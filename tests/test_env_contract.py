"""so101_sim 的对外契约：三个环境 id、无全局命名污染、观测形状。

这些断言是包的对外承诺，重构不得破坏。
"""

import subprocess
import sys
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
DISTRIBUTED_ENVS = [
    "SO101PickPlaceCube40-v1",
    "SO101PickPlaceCube20-v1",
    "SO101PickPlaceCylinder40-v1",
]


def test_three_envs_registered():
    """import 本包后，三个分发环境在 gymnasium 注册表里。"""
    import gymnasium as gym

    import so101_sim  # noqa: F401

    for env_id in DISTRIBUTED_ENVS:
        assert env_id in gym.registry, f"{env_id} 未注册"


def test_no_toplevel_envs_namespace():
    """import so101_sim 不得占用顶层 `envs` 这个名字。

    子进程里跑：顶层 envs 是 RL 圈最易撞名的包名，被我们占了会让学员自己的
    `envs/` 目录莫名其妙失效，且报错完全看不懂。
    """
    code = "import so101_sim, sys; sys.exit(1 if 'envs' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_no_syspath_injection():
    """包内不得有 sys.path 注入：import 只应定义东西，不应改全局搜索路径。"""
    hits = [
        p for p in (PKG_ROOT / "so101_sim").rglob("*.py")
        if "sys.path.insert" in p.read_text() or "sys.path.append" in p.read_text()
    ]
    assert hits == [], f"仍有 sys.path 注入：{hits}"


@pytest.mark.parametrize("env_id", DISTRIBUTED_ENVS)
def test_reset_step_shapes(env_id):
    """原生入口：obs/reward 首维恒为 num_envs，且在 GPU 上。"""
    import gymnasium as gym
    import torch

    import so101_sim  # noqa: F401

    env = gym.make(env_id, num_envs=2, obs_mode="state", sim_backend="gpu",
                   render_mode="all", domain_randomization=False, max_episode_steps=50)
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == 2
    action = torch.zeros((2, env.unwrapped.action_space.shape[-1]),
                         device=env.unwrapped.device)
    obs, reward, terminated, truncated, _ = env.step(action)
    assert obs.shape[0] == 2 and reward.shape[0] == 2
    env.close()


def test_visual_rl_env_returns_maniskill_vector_env():
    """RL 便利函数返回的是 ManiSkill 标准对象，不是我们自造的类型。

    这是"入口只有两个"的机器判据：RL 侧不引入第三种 API。
    """
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    from so101_sim import visual_rl_env

    env = visual_rl_env("SO101PickPlaceCube40-v1", num_envs=2, image_size=16)
    assert isinstance(env, ManiSkillVectorEnv)
    obs, _ = env.reset(seed=0)
    assert obs["rgb"].shape[0] == 2 and obs["rgb"].shape[1] == 16
    env.close()


def test_state_rl_env_returns_maniskill_vector_env():
    """状态 RL 便利函数同样返回 ManiSkill 标准对象；观测是纯 state 张量，没有 rgb 键。"""
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    from so101_sim import state_rl_env

    env = state_rl_env("SO101PickPlaceCube40-v1", num_envs=2)
    assert isinstance(env, ManiSkillVectorEnv)
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == 2
    env.close()


def test_color_jitter_wrapper_jitters_each_camera_independently():
    """双相机拼接后的 6 通道观测，颜色抖动必须逐相机独立采样。

    构造一个假的 6 通道观测直接喂 `ColorJitterWrapper`，不依赖仿真。
    **两路给完全相同的画面**：这样共享一次采样就必然逐元素相等，独立采样才会不等 ——
    输入相同把「输出不同」唯一地归因到采样参数上，是这个构造存在的理由。

    两条断言：输出仍是 6 通道（分组处理没丢通道）；两路输出不相等。
    """
    import gymnasium as gym
    import torch

    from so101_sim.wrappers import ColorJitterWrapper

    torch.manual_seed(0)
    height = width = 8
    # 两路相机给完全相同的画面：这样如果抖动结果不同，只能是因为两组独立采样了参数。
    single_camera = torch.randint(0, 256, (2, height, width, 3), dtype=torch.uint8)
    six_channel = torch.cat([single_camera, single_camera], dim=-1)

    # gym.ObservationWrapper 只需要 observation_space 与 unwrapped 存在即可构造。
    base_env = gym.Env()
    base_env.observation_space = gym.spaces.Dict({
        "rgb": gym.spaces.Box(low=0, high=255, shape=(height, width, 6), dtype="uint8"),
    })
    wrapper = ColorJitterWrapper(base_env)

    out = wrapper.observation({"rgb": six_channel.clone()})
    assert out["rgb"].shape == six_channel.shape

    top_out = out["rgb"][..., :3]
    wrist_out = out["rgb"][..., 3:]
    # 两路相机输入完全相同，若独立采样几乎必然给出不同的抖动结果；若共享同一次采样，
    # 两路输出会逐元素相等。
    assert not torch.equal(top_out, wrist_out), "两路相机的抖动结果相同，怀疑共享了同一份采样参数"


def test_color_jitter_wrapper_rejects_non_multiple_of_three_channels():
    """通道数不是 3 的整数倍时要报清楚的错，而不是静默错乱观测。"""
    import gymnasium as gym
    import torch

    from so101_sim.wrappers import ColorJitterWrapper

    base_env = gym.Env()
    base_env.observation_space = gym.spaces.Dict({
        "rgb": gym.spaces.Box(low=0, high=255, shape=(8, 8, 4), dtype="uint8"),
    })
    wrapper = ColorJitterWrapper(base_env)
    bad_obs = {"rgb": torch.randint(0, 256, (2, 8, 8, 4), dtype=torch.uint8)}

    with pytest.raises(ValueError):
        wrapper.observation(bad_obs)


def test_no_train_env_module():
    """train_env.TrainEnv 已删除：RL 侧走 wrappers，不走自定义类。"""
    import so101_sim

    assert not hasattr(so101_sim, "make_train_env")
    assert not (Path(so101_sim.__file__).parent / "train_env.py").exists()


# ─────────────────────────────────────────────────────────────────────────────
# 评测口的口径契约：默认要与真机逐通道一致
#
# 真机（lerobot-record 走 so_follower）的口径是**混的**：五个臂关节是度，夹爪是
# 0~100 行程百分比（so_follower 把 gripper 写死为 MotorNormMode.RANGE_0_100）。
# ManiSkill 内部恒为弧度。所以这不是一个单位换算，是逐通道换算。
#
# 夹爪那一维尤其要有测试钉住：度数与百分比的量级恰好撞车（物理行程约 0~100 度），
# 看数值看不出错，只表现为抓取这一环学不动。
# ─────────────────────────────────────────────────────────────────────────────

REAL_ARM_PEAK = 30.0      # 度制下复位位姿的臂关节峰值远超此值；弧度下远低于
MANISKILL_ARM_PEAK = 3.2  # 弧度上限约 π


def _make(unit_convention):
    import gymnasium as gym

    import so101_sim  # noqa: F401

    return gym.make(
        "SO101Sim-v1",
        task="SO101PickPlaceCube40-v1",
        episode_length=50,
        control_mode="pd_joint_pos",
        unit_convention=unit_convention,
    )


def test_real_convention_arm_is_degrees_gripper_is_percent():
    """默认口径：臂关节是度，夹爪落在 0~100 且不是角度值。"""
    import numpy as np

    env = _make("real")
    try:
        obs, _ = env.reset(seed=0)
        arm = np.asarray(obs["agent_pos"])[:5]
        grip = float(np.asarray(obs["agent_pos"])[5])
        assert np.abs(arm).max() > REAL_ARM_PEAK, f"臂关节峰值 {np.abs(arm).max()} 不像度制"
        assert 0.0 <= grip <= 100.0, f"夹爪 {grip} 不在行程百分比的 0~100 内"
        # 动作空间的界也必须跟着换：夹爪那一维应当正好是 0~100
        assert float(env.action_space.low[5]) == pytest.approx(0.0, abs=1e-3)
        assert float(env.action_space.high[5]) == pytest.approx(100.0, abs=1e-3)
    finally:
        env.close()


def test_maniskill_convention_is_untouched_radians():
    """`"maniskill"` 那一档必须是原生弧度，六维都不换算。"""
    import numpy as np

    env = _make("maniskill")
    try:
        obs, _ = env.reset(seed=0)
        pos = np.asarray(obs["agent_pos"])
        assert np.abs(pos).max() < MANISKILL_ARM_PEAK, f"峰值 {np.abs(pos).max()} 不像弧度"
        assert float(env.action_space.high[5]) < MANISKILL_ARM_PEAK
    finally:
        env.close()


def test_action_roundtrips_through_real_convention():
    """按真机口径下发的动作，走完几步后状态要停在下发值附近 —— 逐通道验，含夹爪。

    只验观测口径的话，动作那侧漏换算不会被抓到，而那是最致命的一半：
    臂关节会顶到限位，夹爪会张合反向。
    """
    import numpy as np

    env = _make("real")
    try:
        start, _ = env.reset(seed=0)
        target = np.asarray(start["agent_pos"], dtype=np.float32).copy()
        target[0] += 5.0     # 底座 +5°
        target[5] = 60.0     # 夹爪张到 60%
        target = np.clip(target, env.action_space.low, env.action_space.high)
        for _ in range(20):
            obs, *_ = env.step(target)
        got = np.asarray(obs["agent_pos"], dtype=np.float32)
        assert abs(got[0] - target[0]) < 2.0, f"底座停在 {got[0]}°，目标 {target[0]}°"
        assert abs(got[5] - target[5]) < 5.0, f"夹爪停在 {got[5]}%，目标 {target[5]}%"
    finally:
        env.close()


def test_rejects_unknown_unit_convention():
    """非法口径当场报错，不留到评测出一个低成功率再让人反推。"""
    with pytest.raises(ValueError, match="unit_convention"):
        _make("deg")
