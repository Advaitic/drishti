# rl-corrector — debug log (what didn't work, and why)

An honest, chronological record of the errors, dead-ends, and fixes on the way to a working
corrective-VLM loop. Kept so we can (a) write about it later and (b) not re-learn the same lessons.
Raw proof (logs, frames, reports) lives in `results/`.

The goal of the experiment: a `verifiers` RL environment that trains a **corrective VLM** (Qwen3-VL)
to supervise a frozen **VLA worker** (MolmoAct2-SO100_101) doing SO-101 pick-place in MuJoCo. When the
VLA's calibrated uncertainty spikes, the corrector emits a corrective prompt; verifiable reward = sim
success; train via GRPO on self-hosted prime-rl.

Legend: 🔴 = wrong path we went down · 🟢 = the fix · 💡 = the lesson.

---

## Phase 0 — building the env locally (Stage 0, ~$0)

**🔴 Web-research subagents returned plans, not findings.** Dispatched Explore agents in plan mode;
they came back with "here's how I'd research it" instead of answers.
**🟢** Re-dispatched with explicit "use WebSearch/WebFetch directly, deliver findings, don't plan."

**🔴 FastAPI 422 on nested Pydantic models.** Request models defined inside the endpoint closure
tripped FastAPI's schema generation.
**🟢** Moved all request models to module level.

**🔴 macOS GL-context hang — the nastiest local bug.** Rendering MuJoCo inside an `async` FastAPI
endpoint stalls across *successive* requests (the GL context isn't reentrant across the event loop).
Tests would pass once then hang.
**🟢** Refactored the sim/VLA/UQ into a plain `EpisodeManager` decoupled from HTTP, and drove that
directly in tests instead of over the server. (Linux + EGL on the pod doesn't have this problem.)
💡 Don't render inside the async loop; keep the heavy sim state in a sync object the server thinly wraps.

**🔴 StubVLA stalled the MuJoCo solver.** The deterministic CPU stand-in teleported the cube every
frame; the constraint solver couldn't converge.
**🟢** Place the cube once, cleanly, at episode start.

---

## Phase 1 — renting GPUs & loading MolmoAct2 (transformers hell)

**🔴 SSH into Prime pods failed — pubkey not registered.** The `prime` CLI has no ssh-key command.
**🟢** Register via the API before `pods create`: `POST /api/v1/ssh_keys/ {name, publicKey}` (camelCase).

**🔴 Interruptible A100 self-terminated mid-setup** (datacrunch @ $0.63/hr, twice).
**🟢** Use **massedcompute** (datacenter, stable): A100-80GB $1.23/hr, L40-48GB $0.86/hr. Never use
interruptible for a multi-step bring-up. 4090s were out of stock the whole time.

**🔴 transformers 5.x broke MolmoAct2's raw-HF load.** Two failures: a `video_processing` import that
doesn't exist in the checkpoint's remote code, and the tokenizer dying with `'list' object has no
attribute 'keys'`.
**🟢** Pin **transformers>=4.57,<4.58** (Ai2's tested version). And patch the tokenizer config:
MolmoAct2 ships `extra_special_tokens` as a **list**, but 4.57's fast tokenizer wants a name→token
**dict** (`_patch_tokenizer_config` in `policy_molmoact2.py`).

**🔴 torchvision pulled torch off CUDA.** Installing torchvision alone bumped torch to a CPU build.
**🟢** Install the matched pair from the cu121 index: `torch==2.5.1 torchvision==0.20.1`.
Also needed: `einops`, `protobuf`, `sentencepiece`.

---

## Phase 2 — the LeRobot finetune (getting training to run)

**🔴 numpy tried to build from source and failed** (`ninja` missing) under Python 3.13.
**🟢** Force `uv sync --python 3.12` (3.13 had no prebuilt numpy wheel for the resolver) + `apt install
ninja-build`.

**🔴 `datasets` module not found.** LeRobot's dataset loader is behind an extra.
**🟢** `uv sync --extra molmoact2 --extra dataset`.

**🔴 Training validation error: gripper out of range.** `normalize_gripper` defaulted off, but the
gripper is 0–65° — not in [-1,1], so normalization asserted.
**🟢** `--policy.normalize_gripper=true`.

**🟢 Training then ran clean:** action-expert-only finetune, 1500 steps, batch 16, ~1 h on the A100,
**loss 0.65 → 0.02** (`results/logs/ft_full.log`).

---

## Phase 3 — the finetune "didn't grasp": a 2-hour detour that was a frame bug

This is the big one. The model trained fine but **failed in eval (1/8 success, 0 grasps)**. I chased
the wrong causes before finding the real one.

**🔴 Suspected: cadence mismatch.** Thought the sim was stepping too fast vs the demo cadence.
Tried 18 substeps/action. No change.

**🔴 Suspected: execution horizon.** Thought full-chunk open-loop was drifting. Switched to receding
horizon (`n_action_steps=4`). Marginally better but still ~1/8, 0 real grasps.

**🔴 Suspected: eval obs pipeline (image format / gripper convention).** Plausible but unverified.

**🟢 The teacher-forced diagnostic settled it** (`results/logs/tf_check.log`). Feed each *training*
frame's exact obs through the policy, compare predicted action to the true demo action:

```
MEAN ABS ERR per joint (deg): [ 0.73  38.19  91.98  0.00  0.00  5.50 ]
                               pan   lift   elbow  wristF wristR grip
```

The model **learned** — 4 of 6 joints are near-perfect. But **shoulder_lift (j1) and elbow_flex (j2)
are wildly off** — and those are *exactly* the two joints with the sign-flip/+90°-offset transform
(`JOINT_SIGNS=[1,-1,1,1,1,1]`, `JOINT_OFFSETS=[0,90,90,0,0,0]`). **LeRobot's molmoact2 postprocessor
does not cleanly invert `joint_signs/offsets` for the ACTION output.** So the arm's shoulder + elbow
moved to the wrong angles → it wandered instead of grasping.

**🟢 Empirical frame correction** (from the teacher-forced pairs): `env_j1 = 90 − pred`,
`env_j2 = pred + 90`, others pass through. Re-ran eval → **3/8 success, 2 real grasps** (from 0). Bug
confirmed, finetune vindicated.

💡 **The lesson that would have saved 2 hours: validate the action-frame round-trip *before* any
closed-loop eval.** One teacher-forced check (demo obs in → does the predicted action ≈ the demo
action?) catches a frame/normalization bug in 2 minutes. I validated the training *loss* but never the
inference *round-trip*. Loss going to 0.02 only proves the model fits in *its* frame — not that your
serving code reads that frame correctly.

---

## Phase 4 — wiring calibration with the finetuned VLA (this session's bugs)

**🔴 Recorder crashed: `No module named 'fastapi'`.** The recorder imports a few constants from
`rollout_server.py`, which imports FastAPI at module top — but the lerobot venv didn't have it.
**🟢** Installed `fastapi uvicorn httpx` in the lerobot venv (the rollout *server* needs them on the
pod anyway). Note: uv-created venvs have **no `pip`** — use `uv pip install` (find uv at
`~/.local/bin/uv`; it's not on the non-interactive SSH PATH).

**🔴 SigLIP embedder: `SiglipTokenizer requires the SentencePiece library`.** The RND embedder loads a
SigLIP processor, which pulls a sentencepiece tokenizer.
**🟢** `uv pip install sentencepiece protobuf`.

**🔴 SigLIP embedder: `'BaseModelOutputWithPooling' object has no attribute 'float'`.** In
transformers ≥4.57, `SiglipModel.get_image_features` returns an **output object**, not a tensor (older
versions returned a bare tensor). The code called `.float()` on it directly.
**🟢** Unwrap: `feats = getattr(feats, "pooler_output", feats)` before `.float()`
(`uq/embed_frames.py`). 💡 Version-pinning MolmoAct2 to a *newer* transformers silently changed an
*unrelated* model's return type — cross-dependency drift.

**🔴 Recorder got 0/2 success — the VLA suddenly couldn't grasp.** My first VLA adapter ran
`n_action_steps=1` (re-plan a fresh stochastic chunk every frame, execute only step 0). With
flow-matching noise, that jitters and **never commits to the descent** a grasp needs.
**🟢** Buffered execution (`n_action_steps=4`, matching the eval that worked): `select_action` pulls
from the policy's `_action_queue` and only re-plans every 4 steps → it follows through. Result jumped
to **11/16 (69%)**. And ACE sampling is decoupled via `predict_action_chunk` (fresh noise, does NOT
touch the execution queue), so measuring uncertainty never perturbs the arm.
💡 For a chunked flow-matching VLA, receding-horizon *commitment* (nas≥4) matters as much for success
as it does for cost. nas=1 is not "more responsive," it's "more twitchy."

**🔴 `calibrate --auto` → "no successful runs to calibrate on"** despite 11/16 successes. The recorder
emitted telemetry to `ingest_url` (default `localhost:8010`) — but **no ingest server runs on the
pod**, so the async Emitter's HTTP POSTs silently dropped every event (it catches and counts drops).
The store `data/obs.db` stayed empty (0 runs).
**🟢** Construct a `Store()` and pass it to `start_run(store=...)` — the Emitter writes straight to
sqlite (`store` takes priority over `ingest_url` in `_send`). 💡 "in_process" backend still needs a
concrete sink; a default HTTP URL that points at nothing fails *silently*.

---

## Phase 5 — Stage 2 (corrective prompting) + detector tuning

**🔴 The steerability probe corrupted the corrected rollout.** My first Stage-2 driver called
`vla.predict` two extra times per spike (to measure the action-delta under old vs new instruction) — but
`predict` pops the execution queue, so the probe **desynced the arm's committed trajectory** (baseline
looked like 0/4). **🟢** Rewrote as a clean **prompt sweep**: run each instruction variant as the
full-episode task over the *same* seed-0 positions calibration used (so `base` is a built-in sanity check).
💡 Measuring a policy must not consume the same buffer that drives it.

**🔴 ACE never earned its cost.** Both the N-chunk sampling version (AUC 0.43) and the fast single-forward
action-spread (AUC 0.167) are at/below chance — the action-spread tracks motion decisiveness, not failure.
**🟢** Keep the fast (free) version, let **RND carry the combined AUC (1.0)**, drop the N× sampling.
💡 A detector that's cheap AND weak is fine when a strong one covers it — don't pay for a weak one.

**🔴 SMD looked broken (0.52) — but wasn't.** A second calibration draw gave **0.78** (≈ the SmolVLA 0.8).
💡 With 10-vs-6 runs, per-detector AUC swings a lot with *which* failures you draw; report the range, not one
number. SMD is genuinely borderline here because our failures are vision-detectable (gripper open, cube
loose), not proprioceptive — but it's not dead.

**🔴 Unicode in an inline SSH `python -c` broke the shell** (`Δ`, box-drawing → `unexpected EOF`). **🟢**
Write the script to a file + rsync + run, instead of a heredoc/`-c` with non-ASCII.

**🔴 `select_action` and ACE sampling both draw the flow-noise generator**, so calling `sample()` between
`predict`s (recorder) vs not (driver) produces different executed rollouts. Kept in mind when comparing the
recorder's success rate to a driver's; the fast ACE (no sampling) removes this divergence.

## Phase 6 — standing up prime-rl GRPO (the trainer integration, ~10 gotchas)

**🔴 crusoe pods have proxied SSH only** — no public IP ever populates, so rsync + background procs don't
work. **🟢** Use **datacrunch** (direct IP) or massedcompute. Verify `prime pods status --plain` shows an IP
before doing any setup; if N/A after ~5 min, terminate + switch. 💡 Provider SSH model matters more than price.

**🔴 datacrunch offers no `prime_rl` / pytorch image** (only base `ubuntu_22_cuda_12`). **🟢** Provision the
base image + install everything (torch, prime-rl, the lerobot venv) manually.

**🔴 prime-rl's `install.sh` clones a NESTED copy.** Run inside a cloned `prime-rl/`, it creates
`prime-rl/prime-rl/` with the real venv + populated `deps/` submodules; the outer clone's `deps/` stay empty
(`uv run` fails: "pydantic-config is not a Python project"). **🟢** Use the nested `prime-rl/prime-rl/`.

**🔴 Current lerobot needs transformers 5.x** (`torch_compilable_check` import); our 4.57 pin broke its `eo1`
policy import. **🟢** Keep transformers 5.5.4 — the finetuned checkpoint still loads (the 4.57-era HF-load
gotchas were for the raw-HF path, not the lerobot `MolmoAct2Policy` path).

**🔴 macOS rsync (2.6.9) has no `--info=progress2`** — it silently prints usage and exits (the checkpoint
"upload" did nothing). **🟢** Drop `--info`; background rsync with plain `-az`.

**🔴 Qwen3-VL-2B is not in prime-rl's `MODEL_RENDERER_MAP`** → it'd fall back to a renderer that drops the
image parts. **🟢** Use **Qwen3-VL-4B** (mapped → `Qwen3VLRenderer` handles the spike frames). Config caught
this at parse time (validate configs before launching!). Also: LoRA lives at **`[trainer.model.lora]`**, not
`[model.lora]`, in an RL config.

**🔴 `uvicorn module:app` served 500s** — `app = None` at module level shadows the lazy `__getattr__` that
builds the app, so uvicorn got `None`. **🟢** `uvicorn drishti.sim.rollout_server:build_default_app --factory`.

**🔴 SSH commands with `pkill` / `bash -lc` + a backgrounded server truncate their output** (the bg proc holds
the channel; the tool times out). **🟢** Launch with a direct `nohup CMD > log 2>&1 & echo pid`; kill by
explicit pid (never `pkill` in a compound ssh cmd).

**🔴 Three verifiers-0.1.15.dev API diffs broke every rollout** (our env was written for 0.1.14): (1) each
dataset row's `info` must carry `env_id`; (2) `info['task']` is now RESERVED as a JSON task-payload (our
plain-string task → `json.loads` error) → renamed ours to `vla_task`; (3) `state['final_env_response']` must
be a **list of Message dicts**, not the raw report (iterating the dict yielded str keys → "Invalid item type
str"). **🟢** Patched `vla_corrector.py` for all three (editable install picks them up live). 💡 Pin-relaxing
`verifiers==0.1.14`→`>=0.1.14` to accept prime-rl's version means re-checking the multi-turn API.

**🔴 `RuntimeError: 10 consecutive empty train batches`** — prime-rl enforces a `zero_advantage` batch filter
by default, which drops uniform-reward groups; our early steps were all-fail (uniform) → every batch empty →
crash. **🟢** `[[orchestrator.post_batch_filters]] type="zero_advantage" enforce=false` (same as prime-rl's own
`configs/debug/multimodal.toml`). 💡 A tiny smoke on hard episodes is uniform-reward by nature; don't enforce
zero-advantage dropping until reward variance exists.

💡 **The reward-variance lesson (the real one):** the loop *runs* but only *learns* when advantages are
non-zero, which needs reward variance within each GRPO group. Pure-success reward on a VLA that fails
uniformly gives none; the continuous `uq_reduction` **shaping** is what made 66–100% of rollouts trainable.

## Running cost & hardware notes

- Cumulative Prime spend so far ≈ **$5–6** (well under the $200 cap / $500 budget).
- massedcompute A100-80GB @ $1.23/hr is the stable workhorse. Finetune ~1 h; each 16-episode
  calibration recording ~4–5 min (~$0.10).
- Per-frame VLA inference on the A100: ~0.28 s/forward pass (bf16, 8 flow steps). A calibration
  episode is ~0.9 s/frame with ace-b=3.
- The 11 GB finetuned checkpoint is backed up locally at `data/molmoact2_so101_ft_ckpt/` — the pod can
  be terminated without losing it.
