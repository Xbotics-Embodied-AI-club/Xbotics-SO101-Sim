# so101_sim —— SO-101 机械臂仿真环境

基于 ManiSkill3 / SAPIEN（PhysX GPU 后端）的 SO-101 仿真，物体尺寸、机器人几何与
运动速度都按真机 KIT 标定。**独立包**：不依赖 lerobot 也能用。

配套公开数据集：<https://huggingface.co/datasets/Harrysunshine/so101-sim-pickplace>

## ★ 口径：仿真与真机怎么对齐的

**两套口径。选错不报错，只表现为一个会被误读成「策略没学会」的低成功率。**

| 入口 | 怎么拿 | 状态与动作的口径 |
|---|---|---|
| 原生任务口 | `gym.make("SO101PickPlaceCube40-v1")` | **ManiSkill 原生弧度**，六维都是弧度 |
| lerobot 评测口 | `gym.make("SO101Sim-v1", ...)`，或 lerobot 的 `--env.type=so101_sim` | **默认真机口径**（见下） |
| lerobot 机器人口 | lerobot 的 `--robot.type=so101_sim` | **真机口径**，按关节名的动作字典 |

机器人几何**只有一份** URDF（`robots/kit_assets/`）。两份 URDF 会对同一批关节给出不同限位，
取到哪一份取决于注册了哪个 robot uid，而它静默改变夹爪的行程标度 ——
110° 与 130° 把同一个 44.95° 算成 49.95% 或 42.27%。这条由测试钉住。

### 真机口径是「混的」，不是统一的角度

真机数据由 `lerobot-record` 采集，走 lerobot 的 `so_follower`，而它逐关节配的归一化模式**不一样**：

| 通道 | 真机口径 | 出处 |
|---|---|---|
| `shoulder_pan` … `wrist_roll`（5 个臂关节） | **度** | `SOFollowerConfig.use_degrees` 默认 `True` ⇒ `MotorNormMode.DEGREES` |
| `gripper` | **0~100 行程百分比** | `so_follower` 把它**写死**为 `MotorNormMode.RANGE_0_100`，与 `use_degrees` 无关 |

所以「统一到真机」不是一个单位换算，是**逐通道**换算。评测口默认（`unit_convention="real"`）就这么做：

```
臂关节：  弧度 → 度
夹爪：    弧度 → (deg − deg_lo) / (deg_hi − deg_lo) × 100      # deg_lo/hi 取自夹爪关节自己的限位
```

要原生弧度就传 `unit_convention="maniskill"`（数据产线与直接对着 ManiSkill 写的代码走这一档）。

### 为什么夹爪这一处特别容易错

夹爪的**度数与百分比量级恰好撞车**（物理行程约 0~100 度），所以看数值看不出错。
它只表现为抓取这一环学不动 —— 而抓取往往正是唯一学不会的环节。
校验点：仿真「张开到位」44.95° → 49.95%，真机实测张开 50.6%。

### 已经对齐的通道（逐通道审计过，不是推断）

关节名与顺序 · 五个臂关节的度制 · 臂关节量程（URDF 限位换成度后与真机实测几乎逐关节吻合）·
方向同向 · 相机键 `top`+`wrist` · 分辨率 480×640 · fps 30 · `robot_type=so_follower` · 任务文本。

⚠️ 社区的 SO-101 数据集**不都是这个口径**：官方 `lerobot/svla_so101_pickplace` 是归一化
±100（`action` 恰好触到 ±100.00，那是 clamp 的签名）。所以对齐只能以**你自己那台真机**为基准。

## 三个场景

| 环境 id | 场景 |
|---|---|
| `SO101PickPlaceCube40-v1` | 抓 4cm 方块放进料盒 |
| `SO101PickPlaceCube20-v1` | 抓 2cm 方块放进料盒 |
| `SO101PickPlaceCylinder40-v1` | 抓 4cm 圆柱（立在桌面，抓圆面）放进料盒 |

**一个场景一个环境，就这三个。**不做参数化派生，也不为某种用途另开孪生 ——
入口唯一且确定，是这个包的硬约束。

## 装

```bash
pip install git+https://github.com/Xbotics-Embodied-AI-club/Xbotics-SO101-Sim.git
```

要跑 lerobot 评测的话装我们维护的 lerobot fork 即可，它会把本包一起带上：

```bash
pip install "lerobot[all] @ git+https://github.com/Xbotics-Embodied-AI-club/lerobot.git@main"
```

## 三个入口

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
**依赖方向是单向的**：本包的核心不 import lerobot（只有 `lerobot_robot.py` 这一个子模块用到它），
所以只用入口 1 时根本不需要装 lerobot。

⚠️ 评测一个在**绝对关节角**数据上训出来的策略（含上面那个公开数据集）时必须传
`control_mode="pd_joint_pos"`，否则动作被当成归一化增量，**不报错、只会安静地跑错**，
低成功率会被误读成「策略没学会」。同理 `episode_length` 要装得下轨迹长度。

### 3. lerobot 机器人（把仿真当一台真机）

仿真也登记成一个 lerobot 机器人，于是**驱动真机的那些命令原样能用**。
把真机数据集的 action 灌进仿真：

```bash
lerobot-replay \
    --robot.type=so101_sim \
    --robot.discover_packages_path=so101_sim \
    --robot.video_path=out.mp4 \
    --dataset.repo_id=<真机数据集> --dataset.root=<本地根> --dataset.episode=0
```

这是验证 sim2real 对齐最直接的做法：**同一条循环、同一套按关节名取值的动作字典**，
换到真机就是驱动真机。没有第二份回放实现，也就没有第二套口径。
`--robot.discover_packages_path` 是 lerobot 的插件发现口，本包在被 import 时完成注册 ——
lerobot 侧不需要为此改任何代码。

## 结构

```
so101_sim/
├── envs.py                 三个分发场景，就是全部入口（mixin 组合：双相机 / 真机尺寸 / 可达生成 / 速度包线）
├── tasks/                  任务基类与成功判定（place.py + base_random_env.py）
├── robots/
│   ├── so101_kit_slow.py   真机速度包线版（动作空间压进实测包线）
│   ├── so101_base/so101.py 机器人本体：控制器、keyframe、抓取判定
│   └── kit_assets/         **唯一**那份 URDF + 网格 + 物体
├── lerobot_env.py          入口 2：lerobot 评测口
├── lerobot_robot.py        入口 3：把仿真登记成 lerobot 机器人（配置见 config_lerobot_robot.py）
├── wrappers.py             RL 观测包装 + visual_rl_env / state_rl_env
└── _core.py                各入口共享的 gym.make 内核（防止环境定义漂移）
```

## 拿它做强化学习

三个环境的 reward 与步数预算保持原样，**不为训练另注册孪生环境** —— 入口只有这三个。
训练侧要改 reward 或放宽步数，在自己那边包一层 wrapper，别往这个包里加第二个环境 id。

★ 直接拿分发环境的 reward 训之前先知道这件事：「抓着悬在箱口」得 0.61、
「放手仍在箱口」得 0.82–1.00，而 `success` 也是 1.00 —— **成功是零增量**，
且 success 还额外要求机器人静止。600 步预算下整集回报 **悬停 221 vs 成功 4**。
策略「夹着方块在箱口悬停不放手」不是探索失败，是它算对了。
要它去够成功态，得让成功变成**按步计费的持续加成**（一次性奖励在结构上打不过奖励流），
并让训练侧跑满 horizon 而不因成功终止。

★ 步数也要自己放宽：参考轨迹总帧数中位 368、最长 443，而这里是 400 步 ⇒ 余量仅 8.7%，
最长的专家轨迹本身就超预算；按真机包线重计时后还要涨 1.35 倍。

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
