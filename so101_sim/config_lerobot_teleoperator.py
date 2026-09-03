"""把「回放一份已录数据集」登记成一个 lerobot 遥操器。

用途：让仿真数据集**由 `lerobot-record` 产出**，与真机走同一条命令。

    真机： lerobot-record --robot.type=so101_follower --teleop.type=so101_leader ...
    仿真： lerobot-record --robot.type=so101_sim      --teleop.type=so101_dataset_player ...

两条命令除机器人与遥操设备之外逐字相同，于是编码、字段名、单位、帧率都没有第二个
实现，也就无处走偏。此前仿真那四份数据集出自已退役的手工转换路径，视频编码是
`mpeg4` 而真机是 `av1`，而官方合并逐字比 `features` ⇒ 两份根本合不了。

为什么遥操器是"回放"而不是"专家"：`Teleoperator.get_action()` **不接受观测**
（基类签名如此，它设计成读自己那台硬件）。所以看着物体现算的专家放不进这个接口；
能放进来的是**已经算好的一串动作**。纯运动学专家因此是离线规划、这里负责播。

文件名必须是 `config_lerobot_teleoperator.py`：lerobot 的通用回退
`make_device_from_device_class` 用配置类的模块名推设备类的模块名，候选之一是把
`config_` 前缀去掉 ⇒ `so101_sim.lerobot_teleoperator`。与机器人插件那一对同一个约定。
"""

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("so101_dataset_player")
@dataclass(kw_only=True)
class SO101DatasetPlayerConfig(TeleoperatorConfig):
    """播一串**已经算好的**动作。来源两种，二选一。

    Attributes:
        repo_id: 源数据集标识（回放已录数据集时给）。
        episode: 要回放的集号（同上）。
        root: 源数据集根目录。不给则按 lerobot 的默认位置找。
        actions_path: 一份 `.npy` 的路径（离线规划的轨迹时给）。形状 `(帧数, 6)`，
            列序按 `lerobot_robot.JOINT_NAMES`，单位是真机口径（臂五关节度、
            夹爪 0~100 行程百分比）。

    ★ 两种来源**必须恰好给一个**。都给或都不给会在 `connect()` 抛错，而不是挑一个用 ——
      "这批数据是从哪来的"必须由命令本身唯一确定。

    为什么规划走文件而不是让遥操器自己算：`Teleoperator.get_action()` 不接受观测
    （基类签名如此），看着物体现算的专家放不进这个接口。所以专家离线规划、落一份数组，
    这里只负责按顺序播 —— 播的那一侧不做单位换算、不补维、不重排。
    """

    repo_id: str | None = None
    episode: int | None = None
    root: str | None = None
    actions_path: str | None = None
