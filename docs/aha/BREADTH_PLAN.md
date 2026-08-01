# Breadth campaign — does the system generalize across many tasks?

**Question:** does the winning config (per-task calibration + DINOv2-cls encoder + state-window 4 +
embed-window 2) hold at high AUC across a *wide variety* of tasks, or break on some?

**Design (breadth-first):** a *distribution* of per-task AUCs across many tasks is the proof —
so favour breadth over per-task depth.

- **15 diverse tasks** (3 workers × 5), spanning pick/place, open/close, push/press, slide, insert:
  - W0: pick_and_lift, stack_blocks, open_drawer, push_button, insert_onto_square_peg
  - W1: pick_up_cup, put_item_in_drawer, close_box, reach_and_drag, open_door
  - W2: slide_block_to_target, take_lid_off_saucepan, put_money_in_safe, press_switch, pick_and_lift_small
- **Per task:** 10 successes + **4 reliable modes** (grasp, slip, translation_z, wrong_sequence) × 3
  = ~22 demos/task (~330 total). Reliable modes only — skip retry-hungry wrong_object/rotation.
- **Statistics:** per-task overall AUC has SE ~±0.08 at n≈12 failures; the *distribution across 15
  tasks* is what answers "does it generalize" (loose per-task, tight in aggregate). Per-mode
  tightening (±0.05) is a later depth pass, not needed here.

**Why per-task (not per-theme):** pooling hurts even similar tasks (0.56 pooled vs 0.81/0.70
per-task). Profiles are tiny; at deployment the task is known → load its profile. Per-task wins;
theme/global pooling only matters for zero-shot to unseen tasks (a separate, harder claim).

**Compute:** RTX 6000 Ada pod (10 vCPU, $1.04/hr), 3 parallel workers each on its own Xvfb display
(GL isolation), CUDA torch for GPU embedding. 5h generation watchdog. Ablation-ready capture
(30-D state + forces + raw frames + 3 encoders). ~3–4h + embed; ~$5–8.

**Pipeline** (the literal commands; full detail + install in [`GENERATE.md`](GENERATE.md) — there
are no wrapper shell scripts, the parallelism is one `gen_aha.py` process per worker):

```bash
# 1. generate — one process per worker, each on its own Xvfb display
xvfb-run -a -s "-screen 0 1400x900x24" python scripts/gen_aha.py \
  --tasks pick_and_lift stack_blocks open_drawer push_button insert_onto_square_peg \
  --failtypes grasp slip translation_z wrong_sequence \
  --episodes 10 --fail-episodes 3 --save-path /data/aha_out          # ...W1, W2 likewise

# 2. ingest → Store + SigLIP embeds; raw frames are kept BY DEFAULT (--no-frames suppresses them),
#    which is what makes the encoder swap in step 3 an offline re-embed
PYTHONPATH=. python scripts/ingest_aha.py --data /data/aha_out --cams front wrist --embed-stride 8

# 3. GPU re-embed with the other encoders
PYTHONPATH=. python scripts/embed_offline.py --encoder dinov2-cls  --cams front wrist
PYTHONPATH=. python scripts/embed_offline.py --encoder dinov2-mean --cams front wrist

# 4. pull data/obs.db + data/obs_blobs/ to data/aha_breadth/, then per-task AUC — one run per task
DRISHTI_OBS_DB_URL="sqlite:///data/aha_breadth/obs.db" DRISHTI_BLOB_DIR="data/aha_breadth/obs_blobs" \
  PYTHONPATH=. python scripts/eval_aha_seeds.py --source aha --split episode --train-frac 0.7 --k 8 \
  --embedder dinov2-cls --state-window 4 --embed-window 2 --task <TASK>
```

Then document → TERMINATE pod. **Record the state slice with the table**: omitting `--state-fields`
uses the stored 30-D superset *including* the force channels, and that choice is worth up to 0.4 SMD
AUC on some tasks (see `RESULTS.md`, v2 section — this plan's run did not record it, which cost us a
confounded finding).

**Deliverable:** a per-task AUC table/distribution across the 15 tasks with the winning config —
"hits ≥0.85 on X of 15 tasks; struggles on these, here's why."
