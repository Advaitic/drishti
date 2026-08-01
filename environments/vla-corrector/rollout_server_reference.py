"""Reference rollout server for vla-corrector — the HTTP contract the env calls.

This is a TEMPLATE, not a runnable server: fill in the three plug-in points (your Policy, your Sim, and a
calibrated `uq.Calibrator`) and it becomes a working Layer-1 for any action policy. Launch with:

    uvicorn rollout_server_reference:app --host 0.0.0.0 --port 8100     # then set rollout_url to this

The env only relies on the four endpoints + the report shape below; how you produce them is up to you.
Deps: fastapi, uvicorn (+ your policy/sim stack, + the `[uq]` extra for uq.py).
"""
from __future__ import annotations

import base64
import io
import uuid

from fastapi import FastAPI
from pydantic import BaseModel

# from uq import Calibrator     # your fitted detector (fit on successful rollouts once, load here)

MAX_FRAMES, WARMUP = 110, 5


# ----- plug-in point 1: your policy. Must map (frames, task, state) -> the next action. -----
class Policy:
    def reset(self): ...
    def act(self, frames, task, state): raise NotImplementedError   # -> action (your control vector)
    def set_task(self, task): self.task = task                      # a correction just changes the task


# ----- plug-in point 2: your sim. Must step the policy's action and expose success/geometry/frames. -----
class Sim:
    def reset(self, seed=None, cube_xy=None): raise NotImplementedError
    def state(self): raise NotImplementedError                      # proprio vector (SMD input)
    def frames(self): raise NotImplementedError                     # tuple of RGB arrays (RND + corrector input)
    def step(self, action): raise NotImplementedError
    def success(self) -> bool: raise NotImplementedError
    def progress(self) -> float: return 0.0                         # optional staged progress in [0,1]
    def geometry(self) -> dict: return {}                           # optional, shown to the corrector


class Episode:
    def __init__(self, sim, policy, task):
        self.sim, self.policy, self.task = sim, policy, task
        self.frame = self.n_interventions = 0
        self.done = self.success = False


def _b64(img) -> str:
    from PIL import Image
    import numpy as np
    buf = io.BytesIO(); Image.fromarray(np.asarray(img, "uint8")).save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def make_app(new_policy, new_sim, calibrator, task_default="do the task"):
    app, eps = FastAPI(), {}

    def report(ep: Episode, status: str) -> dict:
        frames = ep.sim.frames()
        u = calibrator.score(ep.sim.state(), frames)               # {smd, rnd, combined, threshold, spike}
        geo = ep.sim.geometry()
        text = (f"[uncertainty spike @ frame {ep.frame}] combined={u['combined']:.2f} "
                f"(thr {u['threshold']:.2f}) smd={u['smd']:.2f} rnd={u['rnd']:.2f}\n"
                f"geometry: {geo}\nEmit ONE short corrective instruction (<=20 words).")
        return {"episode_id": ep.id, "status": status, "success": ep.success,
                "done": status in ("success", "timeout"), "progress": ep.sim.progress(),
                "failure_report": text, "geometry": geo,
                "uncertainty": u, "frames_b64": {f"cam{i}": _b64(f) for i, f in enumerate(frames)}}

    class Start(BaseModel):
        seed: int | None = None; cube_xy: list | None = None; task: str = task_default
    class Run(BaseModel):
        episode_id: str; max_frames: int | None = None
    class Correct(BaseModel):
        episode_id: str; prompt: str
    class Id(BaseModel):
        episode_id: str

    @app.post("/episode/start")
    def start(r: Start):
        sim = new_sim(); sim.reset(r.seed, r.cube_xy)
        pol = new_policy(); pol.reset()
        ep = Episode(sim, pol, r.task); ep.id = f"ep_{uuid.uuid4().hex[:8]}"; eps[ep.id] = ep
        return {"episode_id": ep.id}

    @app.post("/episode/run")
    def run(r: Run):
        ep = eps[r.episode_id]; mf = r.max_frames or MAX_FRAMES
        while ep.frame < mf and not ep.done:
            ep.sim.step(ep.policy.act(ep.sim.frames(), ep.task, ep.sim.state()))
            ep.frame += 1
            if ep.sim.success():
                ep.success = ep.done = True; return report(ep, "success")
            if ep.frame >= WARMUP and calibrator.score(ep.sim.state(), ep.sim.frames())["spike"]:
                return report(ep, "spike")                          # pause for a correction
        ep.done = True; return report(ep, "timeout")

    @app.post("/episode/correct")
    def correct(r: Correct):
        ep = eps[r.episode_id]; ep.task = r.prompt; ep.policy.set_task(r.prompt); ep.n_interventions += 1
        return {"episode_id": ep.id, "accepted": True}

    @app.post("/episode/end")
    def end(r: Id):
        eps.pop(r.episode_id, None); return {"ended": True}

    return app


# Wire your own three pieces and expose the result as `app`, then:
#     uvicorn rollout_server_reference:app --port 8100
#
#     app = make_app(Policy, Sim, Calibrator().fit(success_states, success_frames))
#
# Left commented on purpose: this file is a template, and importing it must not try to construct
# somebody's robot. drishti ships a runnable CPU version of the same contract at
# `drishti.sim.stub_server`, useful for exercising the env end to end before you have a policy.
