"""生成物理基线：reset(seed=0) 后走 10 步固定动作，记末态关节角与物体位姿。

产出 `physics_baseline.json`，交给 `tests/test_physics_regression.py` 逐位比对。
动作序列 `0.02*(i%3-1)` 是确定性的、不含随机数，所以两边可以严格对齐。

**只在物理确实该变的时候重跑它**（换了资产真值、改了控制器上限这类）。为了让红掉的
回归测试变绿而重生成基线，等于把这道门关掉 —— 它防的就是「悄悄换了碰撞体」。
"""

import json
import pathlib

import numpy as np
import torch
import gymnasium as gym

import so101_sim  # noqa: F401  导入即注册

ENVS = ["SO101PickPlaceCube40-v1", "SO101PickPlaceCube20-v1", "SO101PickPlaceCylinder40-v1"]
OUT = pathlib.Path(__file__).resolve().parent / "physics_baseline.json"

base = {}
for env_id in ENVS:
    env = gym.make(env_id, num_envs=1, obs_mode="state", sim_backend="gpu",
                   render_mode="all", domain_randomization=False, max_episode_steps=100)
    env.reset(seed=0)
    u = env.unwrapped
    action = torch.zeros((1, u.single_action_space.shape[-1]), device=u.device)
    for i in range(10):
        action[:] = 0.02 * (i % 3 - 1)
        env.step(action)
    base[env_id] = {
        "qpos": u.agent.robot.get_qpos()[0].cpu().numpy().round(6).tolist(),
        "item_p": u.item.pose.p[0].cpu().numpy().round(6).tolist(),
        "item_q": u.item.pose.q[0].cpu().numpy().round(6).tolist(),
    }
    env.close()
    print(env_id, "ok")

OUT.write_text(json.dumps(base, indent=2))
print("baseline written:", OUT)
