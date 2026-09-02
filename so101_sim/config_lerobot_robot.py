"""把仿真登记成一个 lerobot 机器人，`--robot.type=so101_sim` 即可选中。

登记成机器人之后，把真机数据集灌进仿真走的就是 lerobot 自己那条循环：

    lerobot-replay \\
        --robot.type=so101_sim \\
        --robot.discover_packages_path=so101_sim \\
        --dataset.repo_id=<真机数据集> \\
        --dataset.root=<本地根> \\
        --dataset.episode=0

同一条循环、同一套按关节名取值的动作字典，也就是驱动真机的那条 ——
换句话说「真机 action 驱动仿真」与「真机 action 驱动真机」共用一个驱动。

`--robot.discover_packages_path` 是 lerobot 的插件发现口：它 import 本包并遍历
导入所有子模块，本文件在那时完成注册。所以 `so101_sim/__init__.py` 不必依赖
lerobot，仅用仿真器的调用方装不装 lerobot 都能用。

文件名以 `config_` 开头是 lerobot 的解析约定：它按配置类所在模块推导实现类
所在模块（`config_lerobot_robot` → `lerobot_robot`），改名会让 `--robot.type`
解析得到配置却找不到实现。
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("so101_sim")
@dataclass(kw_only=True)
class SO101SimRobotConfig(RobotConfig):
    """一台由仿真扮演的 SO-101。

    Attributes:
        task: 已注册的场景 id。物体几何与料箱位置由它决定。
        episode_length: 单集步数上限，要装得下被回放的那一集。
        seed: 复位种子，决定物体生成位置。同一个种子给出同一个场景。
        video_path: 给了就把机器人相机看到的画面写成 mp4（在 `disconnect` 时落盘）。
        video_camera: 写进视频的相机名。`None` 表示用名字里带 `top` 的那一路。
        state_log_path: 给了就把每一帧测到的关节位置与成功判据写成 npz
            （`state` 形状 `(帧数, 6)`、`success` 形状 `(帧数,)`）。
            用于离线核对「喂进去的动作」与「量回来的状态」——比对是数据比对，
            驱动仍然是 lerobot 自己那条循环。
        sim_backend: 物理后端。`"physx_cpu"`（默认）在单环境下每控制步 3.24ms、
            带渲染整帧约 8ms ⇒ `lerobot-record` 能精确跑到 30 fps；`"gpu"` 每控制步
            47.96ms（18.8 Hz），墙钟驱动的录制会把轨迹截断。两者接触行为在
            `contact_offset` 声明成 5mm 之后已经一致（8 集最高点差 0.1~1.9mm），
            所以这个开关只影响快慢 —— 但**跨后端复验仍要真的两边各跑一遍**。
        initial_state_path: 给了就在复位后把场景置成这份状态（ManiSkill 的
            `set_state_dict` 格式，json）。对应真机上「把物体摆到记录的位置」——
            回放一段录制的动作，物体不在原处就没有可比性。
            不给则由 `seed` 决定物体位置。
    """

    task: str = "SO101PickPlaceCube40-v1"
    episode_length: int = 1000
    seed: int = 0
    video_path: str | None = None
    video_camera: str | None = None
    state_log_path: str | None = None
    initial_state_path: str | None = None
    sim_backend: str = "physx_cpu"
