"""Rollout server — the pod-side FastAPI wrapping SO101Env + a VLA + the UQ cascade.

The heavy half of the split (see the plan's R1): MuJoCo, the MolmoAct2 VLA and the uncertainty
detectors live here, behind episode-scoped HTTP so the `verifiers` env can be a thin client that
carries no torch/mujoco. One episode runs the VLA closed-loop until its calibrated uncertainty
spikes (pause and surface a failure report), the task succeeds, or it times out; the client then
injects a corrective prompt and resumes the same episode.

    # Stage 0 (CPU, stub VLA, ace-only uncertainty) — what the tests build:
    from drishti.sim.rollout_server import create_app, AceOnlyScorer
    from drishti.sim.stub_vla import StubVLA
    app = create_app(vla_factory=lambda env: StubVLA(env), scorer=AceOnlyScorer(threshold=1.5))

    # Stage 1+ (pod, MolmoAct2, calibrated threshold):
    uvicorn drishti.sim.rollout_server:app --host 0.0.0.0 --port 8100

Endpoints: POST /episode/{start,run,correct,end}, GET /episode/{id}, GET /healthz.
"""

from __future__ import annotations

import base64
import io
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from drishti.sim.so101_env import SO101Env
from drishti.sim.scripted_pick import TRAY_XY, _cube_in_tray, _grasp_config, _is_grasping
from drishti.uq import ace_score

DEG2RAD = np.pi / 180.0
# the two views fed to MolmoAct2 SO-100/101: [scene, wrist]. Stage-1 knob (R9) — swap via
# DRISHTI_VLA_CAMS=side,wrist (or up,side) to probe which view is least OOD, no code change.
CAMS_VLA = tuple(os.environ.get("DRISHTI_VLA_CAMS", "up,wrist").split(","))
WARMUP = 3                           # don't flag a spike in the first few frames (start transient)
ACE_RANGE = (-3.2, 3.2)              # joint-radian range for ACE binning (absolute scale, not [-1,1])
N_SUBSTEPS = 18                      # sim steps per control frame (matches live_fleet)
DEFAULT_TASK = "pick up the cube and place it in the box"


# --------------------------------------------------------------------------------------- scorers
class AceOnlyScorer:
    """Stage-0 uncertainty = the intrinsic ACE signal alone (no calibration artifacts needed).

    ACE is calibration-free (spread of the VLA's own action samples), so it lets the whole
    server/env loop be exercised on CPU before any MolmoAct2 rollouts exist to calibrate on."""

    def __init__(self, threshold: float = 1.5):
        self.threshold = float(threshold)

    def score(self, *, smd_raw=0.0, rnd_raw=0.0, ace_raw=0.0) -> tuple[float, str, dict]:
        return float(ace_raw), "ace", {"smd": 0.0, "rnd": 0.0, "ace": float(ace_raw)}


class CalibScorer:
    """Stage-1+ uncertainty = the calibrated SMD+RND+ACE combined score vs. the conformal threshold.

    Reconstructs the detectors from data/calib/ and uses calibrate.combined_score(); `dominant` is
    the detector with the largest capped-z at the frame."""

    def __init__(self, embedder: Optional[Callable] = None, calib_dir=None):
        from drishti.calibrate import load_calib, CAP, _capped_z
        self.smd, self.rnd, self.norm, self.threshold = (
            load_calib(calib_dir) if calib_dir else load_calib())
        self.embedder = embedder
        self._cap, self._capped_z = CAP, _capped_z

    def score(self, *, smd_raw=0.0, rnd_raw=0.0, ace_raw=0.0) -> tuple[float, str, dict]:
        from drishti.calibrate import combined_score
        comb = combined_score(smd_raw, rnd_raw, ace_raw, self.norm, self._cap)
        zs = {k: self._capped_z(v, self.norm[k][0], self.norm[k][1], self._cap)
              for k, v in (("smd", smd_raw), ("rnd", rnd_raw), ("act", ace_raw))}
        dominant = max(zs, key=zs.get)
        return float(comb), ("ace" if dominant == "act" else dominant), {
            "smd": float(smd_raw), "rnd": float(rnd_raw), "ace": float(ace_raw)}


# --------------------------------------------------------------------------------------- geometry
def _render_pair(env: SO101Env, w: int = 320, h: int = 240) -> tuple[np.ndarray, np.ndarray]:
    return env.render(CAMS_VLA[0], w, h), env.render(CAMS_VLA[1], w, h)


def _b64_jpeg(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _phase(env: SO101Env, grasping: bool) -> str:
    gz = float(env.gripper_tcp()[2])
    cube = env.cube_pos()
    if _cube_in_tray(cube, TRAY_XY):
        return "place"
    if grasping:
        return "carry" if gz > 0.10 else "lift"
    return "descend" if gz < 0.06 else "approach"


def _geometry(env: SO101Env, grasping: bool) -> dict:
    g = env.gripper_tcp().astype(float)
    c = env.cube_pos().astype(float)
    return {
        "gripper_tcp": [round(float(v), 4) for v in g],
        "cube_pos": [round(float(v), 4) for v in c],
        "tray_xy": [float(TRAY_XY[0]), float(TRAY_XY[1])],
        "cube_to_tray_dx": round(float(c[0] - TRAY_XY[0]), 4),
        "cube_to_tray_dy": round(float(c[1] - TRAY_XY[1]), 4),
        "gripper_to_cube": round(float(np.linalg.norm(g - c)), 4),
        "grasping": bool(grasping),
    }


def _progress(env: SO101Env, grasping: bool) -> float:
    """Dense staged task-progress in [0,1] — credits partial success so two FAILING episodes differ by
    how far they got (reached < grasped < lifted < placed). This is the signal that gives GRPO reward
    variance from real progress instead of the noisy uq-reduction shaping. Tracked as a per-episode max."""
    g = env.gripper_tcp().astype(float)
    c = env.cube_pos().astype(float)
    if _cube_in_tray(c, TRAY_XY):
        return 1.0                                   # placed (== success)
    if grasping and float(c[2]) > 0.10:
        return 0.85                                  # lifted the cube clear
    if grasping:
        return 0.60                                  # closed on the cube
    dist = float(np.linalg.norm(g - c))
    if dist < 0.05:
        return 0.30                                  # reached the cube (grasp-ready)
    # smooth approach credit toward 0.30: 0 at >=20cm, ramps up as the gripper nears the cube
    return round(0.30 * (1.0 - min(dist, 0.20) / 0.20), 4)


_MEANING = {
    "smd": "the robot's pose is unusual / off-distribution",
    "rnd": "the camera view is unfamiliar to the policy",
    "ace": "the policy's action samples disagree (it is unsure what to do next)",
}


def _compose_report(rep: dict) -> str:
    u, g = rep["uncertainty"], rep["geometry"]
    dom = u["dominant"]
    return (
        f'[UNCERTAINTY SPIKE @ frame {rep["frame"]} | phase={rep["phase"]}]\n'
        f'Detectors (0..1, threshold {u["threshold"]:.2f}): combined={u["combined"]:.2f}  '
        f'SMD={u["smd"]:.2f}  RND={u["rnd"]:.2f}  ACE={u["ace"]:.2f}  -> dominant: {dom.upper()}\n'
        f'Meaning: {_MEANING.get(dom, "")}.\n'
        f'Geometry: gripper at {g["gripper_tcp"]}; cube at {g["cube_pos"]}; tray at {g["tray_xy"]}.\n'
        f'  gripper->cube = {g["gripper_to_cube"]*100:.1f} cm ; '
        f'cube->tray = {abs(g["cube_to_tray_dx"])*100:.1f},{abs(g["cube_to_tray_dy"])*100:.1f} cm ; '
        f'grasping = {"yes" if g["grasping"] else "no"}.\n'
        f'VLA sees / intends: "{rep.get("vla_reasoning","") or "(no trace)"}"\n'
        f'Task so far: "{rep["task"]}"  | interventions so far: {rep["n_interventions"]}\n'
        f'Emit ONE short corrective instruction to the VLA (<= 20 words). No preamble.'
    )


# --------------------------------------------------------------------------------------- episode
@dataclass
class Episode:
    id: str
    env: SO101Env
    task: str
    frame: int = 0
    phase: str = "approach"
    n_interventions: int = 0
    done: bool = False
    success: bool = False
    last_emb: Optional[np.ndarray] = None
    uq_trace: list = field(default_factory=list)
    last_report: Optional[dict] = None
    best_progress: float = 0.0                        # dense staged progress, max over the episode


# ---- request bodies (module-level so FastAPI introspects them as JSON bodies, not query params) ----
class StartReq(BaseModel):
    seed: Optional[int] = None
    cube_xy: Optional[list] = None
    task: str = DEFAULT_TASK


class RunReq(BaseModel):
    episode_id: str
    max_frames: Optional[int] = None


class CorrectReq(BaseModel):
    episode_id: str
    prompt: str


class IdReq(BaseModel):
    episode_id: str


class EpisodeManager:
    """The episode state machine (sim + VLA + UQ), decoupled from HTTP so it is directly testable.

    `vla_factory(env) -> vla` yields a per-episode VLA exposing `predict(images,task,state)->(H,6)`,
    `sample(images,task,state,n)->(n,H,6)`, `reasoning()->str`. `embedder(up,side)->vec` feeds RND when
    `scorer` is a CalibScorer. Every public method serializes on one lock (a single MuJoCo context + one
    GPU VLA per pod); `create_app` is a thin async wrapper over this."""

    def __init__(self, vla_factory: Callable[[SO101Env], Any], scorer: Any, *,
                 max_frames_default: int = 110, ace_n: int = 6, warmup: int = WARMUP,
                 embedder: Optional[Callable] = None, cube_x: tuple = (0.24, 0.28),
                 cube_y: tuple = (-0.03, 0.03)):
        self.vla_factory = vla_factory
        self.scorer = scorer
        self.max_frames_default = max_frames_default
        self.ace_n = ace_n
        self.warmup = warmup
        self.embedder = embedder
        self.cube_x = cube_x
        self.cube_y = cube_y
        self.episodes: dict[str, Episode] = {}
        self.vlas: dict[str, Any] = {}
        self.lock = threading.Lock()

    def _get(self, eid: str) -> Episode:
        ep = self.episodes.get(eid)
        if ep is None:
            raise KeyError(eid)
        return ep

    def _report(self, ep: Episode, status: str) -> dict:
        env, scorer = ep.env, self.scorer
        grasping = _is_grasping(env)
        comb = ep.uq_trace[-1]["combined"] if ep.uq_trace else 0.0
        per = ep.uq_trace[-1] if ep.uq_trace else {"smd": 0.0, "rnd": 0.0, "ace": 0.0, "dominant": "ace"}
        f0, f1 = _render_pair(env)
        rep = {
            "episode_id": ep.id, "status": status, "frame": ep.frame, "phase": ep.phase,
            "task": ep.task, "n_interventions": ep.n_interventions,
            "done": status in ("success", "timeout"), "success": ep.success,
            "reward": 1.0 if ep.success else 0.0,
            "progress": round(ep.best_progress, 4),
            "geometry": _geometry(env, grasping),
            "uncertainty": {
                "smd": per["smd"], "rnd": per["rnd"], "ace": per["ace"], "combined": comb,
                "threshold": scorer.threshold, "dominant": per.get("dominant", "ace"),
                "margin": round(comb - scorer.threshold, 4),
            },
            "uncertainty_trace": ep.uq_trace,
            "vla_reasoning": self.vlas[ep.id].reasoning() if hasattr(self.vlas[ep.id], "reasoning") else "",
            "frames_b64": {CAMS_VLA[0]: _b64_jpeg(f0), CAMS_VLA[1]: _b64_jpeg(f1)},
        }
        rep["failure_report"] = _compose_report(rep)
        ep.last_report = rep
        return rep

    def _step_once(self, ep: Episode) -> None:
        env, vla, scorer = ep.env, self.vlas[ep.id], self.scorer
        f0, f1 = _render_pair(env)
        images = [f0, f1]
        state_rad = env.state()
        chunk = np.asarray(vla.predict(images, ep.task, state_rad), dtype=float)
        action = chunk.reshape(-1, chunk.shape[-1])[0][:6]           # first step of the chunk
        # uncertainty: fast ACE (single-forward chunk spread, ZERO extra passes) if the VLA exposes it,
        # else sample N chunks (HF VLA). Matches how calibrate fit the 'act' signal (mean|action_head_std|).
        if hasattr(vla, "action_head_std"):
            ace_raw = float(np.mean(np.abs(vla.action_head_std())))
        else:
            ace_raw = float(ace_score(vla.sample(images, ep.task, state_rad, self.ace_n),
                                      value_range=ACE_RANGE))
        smd_raw = rnd_raw = 0.0
        if isinstance(scorer, CalibScorer):
            smd_raw = float(scorer.smd.score(env_deg(state_rad)))
            if self.embedder is not None:
                ep.last_emb = np.asarray(self.embedder(f0, f1), dtype=np.float32)
            if ep.last_emb is not None:
                rnd_raw = float(scorer.rnd.score(ep.last_emb))
        comb, dominant, per = scorer.score(smd_raw=smd_raw, rnd_raw=rnd_raw, ace_raw=ace_raw)
        env.step(action.astype(np.float64), n_substeps=N_SUBSTEPS)
        ep.frame += 1
        grasping = _is_grasping(env)
        ep.phase = _phase(env, grasping)
        ep.best_progress = max(ep.best_progress, _progress(env, grasping))
        ep.uq_trace.append({"frame": ep.frame, "phase": ep.phase, "combined": round(comb, 4),
                            "dominant": dominant, **{k: round(v, 4) for k, v in per.items()}})

    # ---- public API (each serialized; mirrors the HTTP endpoints) ----
    def start(self, seed=None, cube_xy=None, task: str = DEFAULT_TASK) -> dict:
        with self.lock:
            rng = np.random.default_rng(seed)
            cxy = (tuple(cube_xy) if cube_xy is not None
                   else (float(rng.uniform(*self.cube_x)), float(rng.uniform(*self.cube_y))))
            env = SO101Env()
            _grasp_config(env, cxy)                        # place cube, lock wrist top-down, settle
            eid = f"ep_{uuid.uuid4().hex[:8]}"
            self.episodes[eid] = Episode(id=eid, env=env, task=task)
            self.vlas[eid] = self.vla_factory(env)
            return {"episode_id": eid, "task": task, "phase": "approach"}

    def run(self, episode_id: str, max_frames: Optional[int] = None,
            stop_on_spike: bool = True) -> dict:
        with self.lock:
            ep = self._get(episode_id)
            mf = max_frames or self.max_frames_default
            if ep.done:
                return self._report(ep, "success" if ep.success else "timeout")
            while ep.frame < mf:
                self._step_once(ep)
                if _cube_in_tray(ep.env.cube_pos(), TRAY_XY):
                    ep.success, ep.done = True, True
                    return self._report(ep, "success")
                if stop_on_spike and ep.frame >= self.warmup and ep.uq_trace[-1]["combined"] > self.scorer.threshold:
                    return self._report(ep, "spike")       # pause; the client injects a correction
            ep.done = True
            return self._report(ep, "timeout")

    def correct(self, episode_id: str, prompt: str) -> dict:
        with self.lock:
            ep = self._get(episode_id)
            ep.task = _sanitize(prompt)
            ep.n_interventions += 1
            return {"episode_id": ep.id, "phase": ep.phase, "task": ep.task, "accepted": True}

    def get(self, eid: str) -> dict:
        with self.lock:
            ep = self._get(eid)
            return ep.last_report or self._report(ep, "success" if ep.success else "running")

    def end(self, eid: str) -> dict:
        with self.lock:
            self.episodes.pop(eid, None)
            self.vlas.pop(eid, None)
            return {"episode_id": eid, "ended": True}

    def healthz(self) -> dict:
        return {"ok": True, "scorer": type(self.scorer).__name__,
                "threshold": self.scorer.threshold, "n_open_episodes": len(self.episodes)}


def create_app(vla_factory: Callable[[SO101Env], Any], scorer: Any, **kwargs) -> FastAPI:
    """Wrap an `EpisodeManager` in FastAPI (thin async endpoints), reachable at `app.state.manager`.

    On the pod (Linux + `MUJOCO_GL=egl`) the render-in-endpoint path is fine. On macOS, rendering inside
    the async loop can stall across successive requests, so the Stage-0 tests drive `EpisodeManager`
    directly instead of over HTTP (see tests/test_rollout_server.py)."""
    mgr = EpisodeManager(vla_factory, scorer, **kwargs)
    app = FastAPI(title="drishti rollout-server")
    app.state.manager = mgr

    def _404(eid):
        return HTTPException(404, f"unknown episode {eid}")

    @app.post("/episode/start")
    async def start(req: StartReq):
        return mgr.start(req.seed, req.cube_xy, req.task)

    @app.post("/episode/run")
    async def run(req: RunReq):
        try:
            return mgr.run(req.episode_id, req.max_frames)
        except KeyError:
            raise _404(req.episode_id)

    @app.post("/episode/correct")
    async def correct(req: CorrectReq):
        try:
            return mgr.correct(req.episode_id, req.prompt)
        except KeyError:
            raise _404(req.episode_id)

    @app.get("/episode/{eid}")
    async def get(eid: str):
        try:
            return mgr.get(eid)
        except KeyError:
            raise _404(eid)

    @app.post("/episode/end")
    async def end(req: IdReq):
        return mgr.end(req.episode_id)

    @app.get("/healthz")
    async def healthz():
        return mgr.healthz()

    return app


def env_deg(state_rad) -> np.ndarray:
    """Env joints (rad) -> degrees, the units the SmolVLA-era calibration/SMD was fit in."""
    return np.asarray(state_rad, dtype=float).reshape(-1)[:6] / DEG2RAD


def _sanitize(prompt: str, max_words: int = 40) -> str:
    """Clamp a correction to one short line (guards against prompt-injection into the next report)."""
    first_line = (str(prompt).strip().splitlines() or [""])[0]
    return " ".join(first_line.split()[:max_words]) or "keep going"


# ------------------------------------------------------------ default app (Stage 1+, on the pod)
def build_default_app() -> FastAPI:
    """MolmoAct2 on the pod. `DRISHTI_LEROBOT_CKPT` → use the finetuned LeRobot VLA (buffered receding
    horizon + joint-frame fix + fast ACE); else the zero-shot HF checkpoint. `UNCERTAINTY=ace` scores the
    intrinsic ACE vs `ACE_THRESHOLD`; `UNCERTAINTY=calib` uses the calibrated SMD+RND+ACE combined score
    from data/calib/ (+ a standalone SigLIP RND embedder). Requires CUDA."""
    import os
    ckpt = os.environ.get("DRISHTI_LEROBOT_CKPT")
    if ckpt:
        from drishti.policy_molmoact2_lerobot import load_molmoact2_lerobot
        vla = load_molmoact2_lerobot(ckpt)
    else:
        from drishti.policy_molmoact2 import load_molmoact2
        vla = load_molmoact2()

    def factory(env):
        if hasattr(vla, "reset"):
            vla.reset()                       # one shared VLA, fresh action queue per episode
        return vla

    if os.environ.get("UNCERTAINTY", "ace") == "ace":
        scorer = AceOnlyScorer(threshold=float(os.environ.get("ACE_THRESHOLD", "5.0")))
        return create_app(vla_factory=factory, scorer=scorer)
    from drishti.uq.embed_frames import load_rnd_embedder
    embedder = load_rnd_embedder()
    return create_app(vla_factory=factory, scorer=CalibScorer(embedder=embedder), embedder=embedder)


app: Optional[FastAPI] = None


def __getattr__(name: str):
    # Lazy so `import drishti.sim.rollout_server` (for create_app in tests) doesn't load MolmoAct2.
    if name == "app":
        return build_default_app()
    raise AttributeError(name)
