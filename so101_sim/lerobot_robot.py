"""仿真扮演的 SO-101，接口与真机 follower 一致，可直接喂给 lerobot 的机器人命令。

内部就是评测口 `So101SimEnv`，与 `lerobot-eval` 走的是同一个类 ——
所以「真机 action 驱动仿真」和「策略驱动仿真」不存在两套驱动。

口径按真机：五个臂关节度制、夹爪 0~100 行程百分比，动作是绝对关节位置目标。
动作按**关节名**取值而不是按下标，因为调用方给的是 `{"shoulder_pan.pos": ...}`
这样的字典 —— 按下标取会在特征顺序变化时静默取错关节。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from lerobot.robots import Robot
from lerobot.types import RobotAction, RobotObservation

from so101_sim.config_lerobot_robot import SO101SimRobotConfig
from so101_sim.lerobot_env import So101SimEnv
from so101_sim.robots.so101_base.so101 import SO101

# 关节顺序取自机器人自己的声明，不在这里另抄一份。
JOINT_NAMES = tuple(SO101.arm_joint_names) + tuple(SO101.gripper_joint_names)

# 已交付数据集录的是绝对关节角，所以只能用绝对位置控制器。
# 用默认的归一化增量模式不会报错，只会把绝对角逐维 clip 到 ±1，手臂以包线最大速度
# 朝错误方向走 —— 表现成一个会被误读成「策略没学会」的低成功率。
CONTROL_MODE = "pd_joint_pos"


class SO101SimRobot(Robot):
    """一台由仿真扮演的 SO-101，`get_observation` / `send_action` 与真机同形。"""

    config_class = SO101SimRobotConfig
    name = "so101_sim"

    def __init__(self, config: SO101SimRobotConfig):
        super().__init__(config)
        self.config = config
        self._env: So101SimEnv | None = None
        self._obs: dict[str, Any] | None = None
        self._frames: list[np.ndarray] = []
        self._states: list[np.ndarray] = []
        self._success: list[bool] = []

    @property
    def observation_features(self) -> dict:
        """六个关节位置 + 每路相机的画面形状。"""
        feats: dict[str, Any] = {f"{name}.pos": float for name in JOINT_NAMES}
        if self._env is not None:
            shape = (self._env.observation_height, self._env.observation_width, 3)
            for cam in self._env._camera_names:
                feats[cam] = shape
        return feats

    @property
    def action_features(self) -> dict:
        """六个关节的绝对位置目标，真机口径。"""
        return {f"{name}.pos": float for name in JOINT_NAMES}

    @property
    def is_connected(self) -> bool:
        return self._env is not None

    def connect(self, calibrate: bool = True) -> None:
        """建场景并复位到开机位姿。

        Args:
            calibrate: 忽略 —— 仿真的零点由 URDF 给定，没有可标定的电机。
        """
        if self._env is not None:
            raise RuntimeError("已经连上了；重复 connect 会丢掉当前这一集的状态")
        self._env = So101SimEnv(
            task=self.config.task,
            obs_type="pixels_agent_pos",
            control_mode=CONTROL_MODE,
            episode_length=self.config.episode_length,
            unit_convention="real",
        )
        self._obs, _ = self._env.reset(seed=self.config.seed)
        if self.config.initial_state_path:
            self._obs = self._apply_initial_state(Path(self.config.initial_state_path))
        self._frames = []
        self._states = []
        self._success = []

    @property
    def is_calibrated(self) -> bool:
        """恒为真：零点由 URDF 给定。"""
        return True

    def calibrate(self) -> None:
        raise NotImplementedError(
            "仿真没有可标定的电机 —— 零点由 URDF 的关节限位给定。"
            "真机与仿真的零点若有系统偏差，那是真机标定的事，不在这一侧补。"
        )

    def configure(self) -> None:
        """无需配置：控制器与限位都由场景与 URDF 定好。"""

    def get_observation(self) -> RobotObservation:
        """当前关节位置与各路相机画面。

        Returns:
            `{"<关节>.pos": 度或百分比}` 加上 `{相机名: (H,W,3) uint8}`。

        Raises:
            RuntimeError: 还没 `connect`。
        """
        if self._env is None or self._obs is None:
            raise RuntimeError("还没 connect，没有观测可取")
        pos = np.asarray(self._obs["agent_pos"], dtype=np.float64).reshape(-1)
        obs: dict[str, Any] = {
            f"{name}.pos": float(pos[i]) for i, name in enumerate(JOINT_NAMES)
        }
        obs.update(self._obs["pixels"])
        if self.config.video_path:
            self._frames.append(self._pick_frame(self._obs["pixels"]))
        if self.config.state_log_path:
            self._states.append(pos.astype(np.float32))
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        """把一帧绝对关节位置目标发下去，走一步仿真。

        Args:
            action: `{"<关节>.pos": 值}`，真机口径。

        Returns:
            实际下发的那一份（与入参同形）。

        Raises:
            RuntimeError: 还没 `connect`。
            KeyError: 缺关节 —— 缺一维就按缺省值补会静默改变轨迹。
        """
        if self._env is None:
            raise RuntimeError("还没 connect，无法下发动作")
        missing = [f"{n}.pos" for n in JOINT_NAMES if f"{n}.pos" not in action]
        if missing:
            raise KeyError(f"动作里缺这些关节：{missing}；收到的键是 {sorted(action)}")
        vec = np.array([float(action[f"{n}.pos"]) for n in JOINT_NAMES], dtype=np.float32)
        self._obs, _, _, _, info = self._env.step(vec)
        if self.config.state_log_path:
            self._success.append(bool(info.get("is_success", False)))
        return {f"{n}.pos": float(vec[i]) for i, n in enumerate(JOINT_NAMES)}

    def disconnect(self) -> None:
        """收掉场景；要了视频的话在这里落盘。"""
        if self._env is None:
            return
        if self.config.video_path and self._frames:
            self._write_video()
        if self.config.state_log_path and self._states:
            path = Path(self.config.state_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            n = min(len(self._states), len(self._success)) if self._success else len(self._states)
            np.savez(path, state=np.stack(self._states)[:n],
                     success=np.asarray(self._success[:n], dtype=bool))
        self._env.close()
        self._env = None
        self._obs = None

    def _apply_initial_state(self, path: Path) -> dict[str, Any]:
        """把场景置成给定状态，并回读一份观测。

        Args:
            path: ManiSkill `get_state_dict()` 的 json 形式。

        Returns:
            置位之后的观测。

        Raises:
            RuntimeError: 置位后渲染没跟上 —— GPU 后端改状态必须显式刷新，
                否则画面还是旧的而数值已经变了。
        """
        import json

        import torch

        raw = json.loads(path.read_text())

        def to_tensor(node):
            if isinstance(node, dict):
                return {k: to_tensor(v) for k, v in node.items()}
            return torch.as_tensor(np.asarray(node, dtype=np.float32), device=self._env._env.device)

        inner = self._env._env.unwrapped
        inner.set_state_dict(to_tensor(raw))
        # GPU 后端：改完状态要走一遍 apply/fetch，画面与后续读数才是新状态的。
        if inner.gpu_sim_enabled:
            inner.scene._gpu_apply_all()
            inner.scene.px.gpu_update_articulation_kinematics()
            inner.scene._gpu_fetch_all()
        return self._env._format_raw_obs(inner.get_obs())

    def _pick_frame(self, pixels: dict[str, np.ndarray]) -> np.ndarray:
        """挑一路相机的当前帧。

        Args:
            pixels: 相机名到画面的字典。

        Returns:
            `(H, W, 3)` uint8。

        Raises:
            KeyError: 指定的相机名不在这一场景里。
        """
        if self.config.video_camera:
            if self.config.video_camera not in pixels:
                raise KeyError(
                    f"没有相机 {self.config.video_camera!r}；本场景的相机是 {sorted(pixels)}"
                )
            return np.asarray(pixels[self.config.video_camera])
        tops = [k for k in sorted(pixels) if "top" in k]
        return np.asarray(pixels[tops[0] if tops else sorted(pixels)[0]])

    def _write_video(self) -> None:
        """把攒下的帧按场景帧率写成 mp4。"""
        import imageio.v2 as imageio

        path = Path(self.config.video_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fps = So101SimEnv.metadata["render_fps"]
        imageio.mimwrite(path, self._frames, fps=fps, macro_block_size=1)
