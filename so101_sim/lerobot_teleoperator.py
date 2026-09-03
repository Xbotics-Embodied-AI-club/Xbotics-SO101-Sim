"""播放型遥操器：把一串**已经算好的**动作逐帧吐给 `lerobot-record`。

动作来源二选一（见 `config_lerobot_teleoperator.py`）：一份已录数据集的某一集，
或一份离线规划落盘的 `.npy`。后者是脚本化专家用的口子 —— `Teleoperator.get_action()`
不接受观测，看着物体现算的专家放不进这个接口，所以专家离线规划、这里只负责播。

这里只做一件事：**按顺序把动作发出去**，不做单位换算、不补维、不重排 ——
那三件事任何一件放进来，仿真数据就又多了一个口径。

★ 关节顺序按**名字**取，不按列序。源数据集的 `action` 有 `names`，本包有 `JOINT_NAMES`，
  两者名字集合必须相同，否则直接报错。按列序取在名字顺序变了的时候不报错，只是把
  肩转的值发给了肘弯 —— 那正是"静默改变轨迹"。
"""

from typing import Any

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction

from .config_lerobot_teleoperator import SO101DatasetPlayerConfig
from .lerobot_robot import JOINT_NAMES


class SO101DatasetPlayer(Teleoperator):
    """把一集已录动作当作"遥操输入"播出去。"""

    config_class = SO101DatasetPlayerConfig
    name = "so101_dataset_player"

    def __init__(self, config: SO101DatasetPlayerConfig):
        super().__init__(config)
        self.config = config
        self._actions: np.ndarray | None = None
        self._cursor = 0

    @property
    def action_features(self) -> dict:
        """六个关节的绝对位置目标，真机口径 —— 与机器人那侧同一份名字。"""
        return {f"{name}.pos": float for name in JOINT_NAMES}

    @property
    def feedback_features(self) -> dict:
        """不吃反馈：回放是开环的。"""
        return {}

    @property
    def is_connected(self) -> bool:
        return self._actions is not None

    def connect(self, calibrate: bool = True) -> None:
        """把要播的动作全部读进内存 —— 来自已录数据集或一份规划好的 `.npy`。

        Args:
            calibrate: 基类接口要求，回放没有标定这回事，忽略。

        Raises:
            ValueError: 两种来源不是恰好给一个；或源数据集的关节名与本包的对不上；
                或 `.npy` 的形状不是 `(帧数, 6)`。
        """
        from_dataset = self.config.repo_id is not None
        from_file = self.config.actions_path is not None
        if from_dataset == from_file:
            raise ValueError(
                "`repo_id` 与 `actions_path` 必须恰好给一个 —— "
                f"现在 repo_id={self.config.repo_id!r} actions_path={self.config.actions_path!r}。"
                "两个都给就说不清这批数据从哪来。")
        if from_file:
            actions = np.load(self.config.actions_path)
            want = len(JOINT_NAMES)
            if actions.ndim != 2 or actions.shape[1] != want:
                raise ValueError(
                    f"{self.config.actions_path} 的形状是 {actions.shape}，"
                    f"要的是 (帧数, {want})，列序按 JOINT_NAMES")
            self._actions = np.asarray(actions, dtype=np.float64)
            self._cursor = 0
            return

        dataset = LeRobotDataset(self.config.repo_id, root=self.config.root,
                                 episodes=[self.config.episode])
        names = dataset.features["action"]["names"]
        want = [f"{n}.pos" for n in JOINT_NAMES]
        if sorted(names) != sorted(want):
            raise ValueError(f"源数据集的关节名与本包不一致：\n  源 {names}\n  本包 {want}")
        order = [names.index(n) for n in want]
        # 取列的写法与 `lerobot_replay.py` 一致（`dataset.select_columns("action")`
        # 再逐帧取），那条路径是官方回放在用的，不另找一种。
        column = dataset.select_columns("action")
        self._actions = np.asarray(
            [column[i]["action"] for i in range(dataset.num_frames)], dtype=np.float64)[:, order]
        self._cursor = 0

    @property
    def is_calibrated(self) -> bool:
        """回放没有标定这回事，恒为真。"""
        return True

    def calibrate(self) -> None:
        """回放没有标定这回事。"""

    def configure(self) -> None:
        """回放没有需要配置的硬件。"""

    def get_action(self) -> RobotAction:
        """下一帧动作。

        源动作放完之后**保持最后一帧**：`lerobot-record` 按 `episode_time_s` 计时，
        比源集稍长时手臂停在原处，是这里唯一说得通的行为。编排器会把
        `episode_time_s` 设成正好等于源集时长，所以正常情况下用不到这个兜底。

        Returns:
            `{"<关节>.pos": 值}`，真机口径。

        Raises:
            RuntimeError: 还没 `connect`。
        """
        if self._actions is None:
            raise RuntimeError("还没 connect，没有可回放的动作")
        row = self._actions[min(self._cursor, len(self._actions) - 1)]
        self._cursor += 1
        return {f"{name}.pos": float(row[i]) for i, name in enumerate(JOINT_NAMES)}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """回放是开环的，不吃反馈。"""

    def disconnect(self) -> None:
        """丢掉动作缓冲。"""
        self._actions = None
        self._cursor = 0

    @property
    def n_frames(self) -> int:
        """源集的帧数。编排器按它算 `episode_time_s`。

        Raises:
            RuntimeError: 还没 `connect`。
        """
        if self._actions is None:
            raise RuntimeError("还没 connect，不知道帧数")
        return len(self._actions)
