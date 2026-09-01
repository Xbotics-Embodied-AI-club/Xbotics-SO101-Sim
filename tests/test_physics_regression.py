"""物理不许被动过：与入库的基线逐位比对。

**这道门专抓一类静默失效**：URDF 里的网格路径是相对路径，搬一个目录就会加载到错的
碰撞体 —— 画面看着还正常，抓取深度全变，没有任何一步报错。搬文件、改 import、
换依赖版本都可能触发它，而别的测试都抓不住。

基线由 `data/gen_physics_baseline.py` 生成：`reset(seed=0)` 后走 10 步确定性动作
`0.02 * (i % 3 - 1)`，记末态关节角 `qpos` 与物体位姿 `item_p` / `item_q`。
基线文件缺任何一个环境或键，都视为基线本身出了问题，必须让测试失败而不是悄悄跳过 ——
「扫描目标不存在」被读成「查过了」正是这道门要防的形状。

基线与生成脚本随测试一起入库（`data/`），换台机器重新 clone 之后这道门照样跑得起来。
"""

import json
from pathlib import Path

import numpy as np
import pytest

BASELINE_PATH = Path(__file__).resolve().parent / "data/physics_baseline.json"

ENV_IDS = [
    "SO101PickPlaceCube40-v1",
    "SO101PickPlaceCube20-v1",
    "SO101PickPlaceCylinder40-v1",
]


def _load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        pytest.fail(
            f"缺少物理基线文件：{BASELINE_PATH}。"
            "这道门是「物理没被动过」的唯一判据，没有基线不能跳过，必须先补齐。"
        )
    return json.loads(BASELINE_PATH.read_text())


def _align_quaternion_sign(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """四元数 q 与 -q 表示同一个旋转，逐位比对前先对齐符号，避免假失败。"""
    if np.dot(actual, expected) < 0:
        return -actual
    return actual


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_matches_prerefactor_baseline(env_id):
    """跑十步确定性动作，末态与基线逐位比。

    Args:
        env_id: 三个分发环境之一。

    比较四元数前先对齐 q/-q 符号：两者表示同一个旋转，不对齐会得到假失败。
    """
    baseline = _load_baseline()
    assert env_id in baseline, f"基线文件里缺 {env_id} 这个环境的记录"
    expected = baseline[env_id]
    for key in ("qpos", "item_p", "item_q"):
        assert key in expected, f"{env_id} 的基线记录缺 {key}"

    import gymnasium as gym
    import torch

    import so101_sim  # noqa: F401  导入即注册

    env = gym.make(env_id, num_envs=1, obs_mode="state", sim_backend="gpu",
                    render_mode="all", domain_randomization=False, max_episode_steps=100)
    try:
        env.reset(seed=0)
        u = env.unwrapped
        action = torch.zeros((1, u.single_action_space.shape[-1]), device=u.device)
        for i in range(10):
            action[:] = 0.02 * (i % 3 - 1)  # 与 baseline 生成时完全相同的动作序列
            env.step(action)

        actual_qpos = u.agent.robot.get_qpos()[0].cpu().numpy()
        actual_p = u.item.pose.p[0].cpu().numpy()
        actual_q = u.item.pose.q[0].cpu().numpy()

        np.testing.assert_allclose(
            actual_qpos, expected["qpos"], atol=1e-5,
            err_msg=f"{env_id} 关节角与基线不符",
        )
        np.testing.assert_allclose(
            actual_p, expected["item_p"], atol=1e-5,
            err_msg=f"{env_id} 物体位置与基线不符",
        )
        aligned_q = _align_quaternion_sign(actual_q, np.asarray(expected["item_q"]))
        np.testing.assert_allclose(
            aligned_q, expected["item_q"], atol=1e-5,
            err_msg=f"{env_id} 物体姿态与基线不符（已做 q/-q 符号对齐）",
        )
    finally:
        env.close()
