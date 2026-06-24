"""Determine SmolVLA's action output units: raw predict_action_chunk vs postprocess,
compared to the dataset action (degrees). Tells the driver which to execute."""

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from drishti.drivers.replay import _obs_batch
from drishti.policy_smolvla import load_smolvla

ds = LeRobotDataset("lerobot/svla_so101_pickplace")
policy, preprocess, postprocess, device = load_smolvla()
item = ds[0]
proc = preprocess(_obs_batch(item, device))

with torch.inference_mode():
    chunk = policy.predict_action_chunk(proc, noise=None)
arr = chunk.detach().float().cpu().numpy()
flat = arr.reshape(-1, arr.shape[-1])
print("raw chunk shape:", arr.shape)
print("raw  min:", np.round(flat.min(0)[:6], 2))
print("raw  max:", np.round(flat.max(0)[:6], 2))

try:
    post = postprocess(chunk if chunk.ndim == 2 else chunk[0])
    pa = post.detach().float().cpu().numpy() if hasattr(post, "detach") else np.asarray(post)
    pf = pa.reshape(-1, pa.shape[-1])
    print("post min:", np.round(pf.min(0)[:6], 2))
    print("post max:", np.round(pf.max(0)[:6], 2))
except Exception as e:
    print("postprocess failed:", repr(e))

print("dataset action[0]:", np.round(np.asarray(item["action"])[:6], 2))
print("=> the one whose range ~matches dataset degrees (0-100ish) is the executable action")
