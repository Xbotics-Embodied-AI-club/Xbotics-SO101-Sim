# so101_sim —— SO-101 机械臂仿真环境

基于 ManiSkill3 / SAPIEN（PhysX GPU 后端）的 SO-101 仿真，物体尺寸、机器人几何与
运动速度都按真机 KIT 标定。**独立包**：不依赖 lerobot 也能用。

配套公开数据集：<https://huggingface.co/datasets/Harrysunshine/so101-sim-pickplace>

## 三个场景

| 环境 id | 场景 |
|---|---|
| `SO101PickPlaceCube40-v1` | 抓 4cm 方块放进料盒 |
| `SO101PickPlaceCube20-v1` | 抓 2cm 方块放进料盒 |
| `SO101PickPlaceCylinder40-v1` | 抓 4cm 圆柱（立在桌面，抓圆面）放进料盒 |

一个场景一个环境，不做参数化派生。每个场景另有一个 `...Train-v1` 孪生环境供 RL 训练用
（改 reward 与步数预算，物理不变，见下）。

## 装

```bash
pip install git+https://github.com/Xbotics-Embodied-AI-club/Xbotics-SO101-Sim.git
```

要跑 lerobot 评测的话装我们维护的 lerobot fork 即可，它会把本包一起带上：

```bash
pip install "lerobot[all] @ git+https://github.com/Xbotics-Embodied-AI-club/lerobot.git@xbotics"
```

## 两个入口

### 1. 原生（ManiSkill 批量环境）

数据生产与 RL 都走这个。观测是 **GPU 上的 torch tensor**，首维恒为 `num_envs`：

```python
import gymnasium as gym
import so101_sim  # import 即注册

env = gym.make("SO101PickPlaceCube40-v1", num_envs=64, obs_mode="state",
               sim_backend="gpu", render_mode="all")
obs, _ = env.reset(seed=0)      # obs.shape == (64, ...)，在 cuda 上
```

视觉 RL 另有一个便利构造器（降采样 16px + 颜色抖动 + 向量化），返回的仍是
ManiSkill 标准的 `ManiSkillVectorEnv`：

```python
from so101_sim import visual_rl_env
env = visual_rl_env("SO101PickPlaceCube40-v1", num_envs=1024, image_size=16)
```

### 2. lerobot 评测（标准单环境 gym.Env）

`So101SimEnv` 把批量 tensor 观测转成 lerobot 约定的 numpy 格式
（`{"agent_pos": ..., "pixels": {"top": ..., "wrist": ...}}`，两路相机都给）：

```bash
lerobot-eval --env.type=so101_sim --env.task=SO101PickPlaceCube40-v1 --eval.n_episodes=20
```

`--env.type=so101_sim` 这个选项**上游 lerobot 没有**，由我们维护的 fork
（[Xbotics-Embodied-AI-club/lerobot](https://github.com/Xbotics-Embodied-AI-club/lerobot)）注册。
**依赖方向是单向的**：本包不 import lerobot，是 lerobot 认识本包 —— 所以只用入口 1 时
根本不需要装 lerobot。

⚠️ 评测一个在**绝对关节角**数据上训出来的策略（含上面那个公开数据集）时必须传
`control_mode="pd_joint_pos"`，否则动作被当成归一化增量，**不报错、只会安静地跑错**，
低成功率会被误读成「策略没学会」。同理 `episode_length` 要装得下轨迹长度。

## 结构

```
so101_sim/
├── envs.py                 三个分发场景 + 三个 RL 训练孪生（mixin 组合：双相机 / 真机尺寸 / 可达生成 / 速度包线）
├── tasks/                  任务基类与成功判定（place.py + base_random_env.py）
├── robots/
│   ├── so101_kit.py        KIT 版机器人（含底板、型材、两个相机支架）
│   ├── so101_kit_slow.py   真机速度包线版
│   ├── so101_base/         裸臂 SO101 本体与网格
│   └── kit_assets/         KIT URDF + 网格 + 物体
├── lerobot_env.py          入口 2：lerobot 评测口
├── wrappers.py             RL 观测包装 + visual_rl_env / state_rl_env
└── _core.py                两个入口共享的 gym.make 内核（防止环境定义漂移）
```

## RL 训练孪生环境

三个 `...Train-v1` 只改 reward 与步数预算，**物理与分发环境逐位相同**。为什么必须分开：
分发环境的 reward 里「抓着悬在箱口」得 0.61、「放手仍在箱口」得 0.82–1.00，而 `success`
也是 1.00 —— 成功是零增量，且 success 还额外要求机器人静止。600 步预算下整集回报
**悬停 221 vs 成功 4**，策略「夹着方块在箱口悬停不放手」是它算对了，不是探索失败。

分发环境的 reward 与步数保持原样不动：它们的物理行为被 `tests/test_physics_regression.py`
逐位钉住，而且已经产出了公开数据集。

## 自检

```bash
pip install -e ".[dev]"
pytest
```

两组门：`tests/test_env_contract.py`（对外契约：无 `sys.path` 注入、不占顶层 `envs` 命名空间）与
`tests/test_physics_regression.py`（物理不变：三环境 `qpos`/`item_p`/`item_q` 逐位比对基线）。

**搬资产文件后必须跑物理回归** —— URDF 里网格是相对路径，搬错目录会静默加载错的碰撞体，
画面正常但抓取深度全变。

## 来源与许可

MIT（见 [`LICENSE`](LICENSE)）。`tasks/` 与 `robots/so101_base/` 最初取自
[squint](https://github.com/aalmuzairee/squint)（MIT，© 2026 Abdulaziz Almuzairee），
自 2026-08-23 起分叉自维护 —— 署名见 [`NOTICE`](NOTICE)，分叉范围见
[`so101_sim/tasks/UPSTREAM.md`](so101_sim/tasks/UPSTREAM.md)。
