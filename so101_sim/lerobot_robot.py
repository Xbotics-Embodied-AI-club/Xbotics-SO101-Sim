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
from so101_sim.envs import CAMERA_HEIGHT, CAMERA_NAMES, CAMERA_WIDTH
from so101_sim.lerobot_env import So101SimEnv
from so101_sim.robots.so101_base.so101 import SO101

# 关节顺序取自机器人自己的声明，不在这里另抄一份。
JOINT_NAMES = tuple(SO101.arm_joint_names) + tuple(SO101.gripper_joint_names)

# 已交付数据集录的是绝对关节角，所以只能用绝对位置控制器。
# 用默认的归一化增量模式不会报错，只会把绝对角逐维 clip 到 ±1，手臂以包线最大速度
# 朝错误方向走 —— 表现成一个会被误读成「策略没学会」的低成功率。
CONTROL_MODE = "pd_joint_pos"

# 舵机编码器的一个计数（度）。**是 360/4095 不是 360/4096** ——
#
# **真机的关节读数落在这个栅格上，仿真的必须也落在上面。** 实测
# `pick_up_a_cube` 300 集：`min|Δ| = 0.087906°`，非零逐帧差的 99.90~100.0% 是它的整数倍；
# 而未量化的仿真 `min|Δ|` 只有 7.5e-05°。一行 `np.diff(state)/0.087891` 就能把两份
# 数据分开 —— 这是「看得出不是真机录的」里最直接的一条。
#
# 量化还顺带补上另一条痕迹：真机的 state 会**停住**（相邻帧完全相等的比例 37.4~76.6%，
# 最长同值段中位 42~72 帧，每集末尾 10 帧标准差精确为 0），因为动得比一个计数慢时
# 读数不变。未量化的仿真永远在漂（相邻帧相等 0.0~1.2%，最长同值段 1 帧）。
# lerobot 自己就是这么换算的：`motors/motors_bus.py:858` 取 `max_res = 分辨率 − 1 = 4095`，
# 再 `(val − mid) * 360 / max_res`。最小二乘从真机数据直接反解（不预设候选）也给出
# 0.08791209：两个任务 × action/state × 5 关节共 20 组，八位小数全一致。
# 判据要收紧才分得出来：容差 0.02 时 360/4096 与 360/4095 都能过（99.96% vs 100%），
# 收到 0.002 才分开（57.6% vs 100.0%）。
ENCODER_STEP_DEG = 360.0 / 4095.0

def gripper_step_pct():
    """夹爪那一路的编码器量子（行程百分点/格），**从本机 URDF 现算**。

    Returns:
        一个计数占满行程的百分比。

    夹爪的 0~100 是按 URDF 行程定义的（`so101.gripper_limit_rad`），而同一根轴上
    也是那颗 12 位舵机，所以量子 = 100 ÷（行程占满量程的格数）：
    URDF 行程 110.00° ⇒ 110/360×4095 = 1251.3 格 ⇒ 0.079920 %/格。

    ★ **这是一个数，不是一张按任务查的表。** 曾按场景查表（cube 用 100/1477、
      cylinder 用 100/1519，抄各自对标真机那一台）—— 那等于说同一台机器夹不同
      物体时物理标定会变，而且每加一个任务就要加一行。标定是**机器**的属性，
      不是任务的属性。

    ★ **也不该抄某一台真机的跨度。** 夹爪百分比是标定相对量：真机各台自己就不一样
      （主臂侧 1260 / 1268 / 1279，彼此差 1.5%），仿真这台 1251 落在这个散布里
      （离最近的一台 0.7%）。仿真是「又一台机器」，用自己的尺子才自洽 ——
      改抄真机尺子那次已被高尔夫球证伪（刚体标准件 42.67mm 被预测成 53.7mm）。

    ★ 现读不写死：URDF 换了这里跟着变，与 `so101.gripper_limit_rad` 同一条规矩。
    """
    from so101_sim.robots.so101_base.so101 import gripper_limit_rad

    low, high = gripper_limit_rad()
    return 100.0 / (np.degrees(high - low) / ENCODER_STEP_DEG)


class SO101SimRobot(Robot):
    """一台由仿真扮演的 SO-101，`get_observation` / `send_action` 与真机同形。"""

    config_class = SO101SimRobotConfig
    # ★`name` 决定写进数据集的 `robot_type`（`Robot.__init__` 里 `self.robot_type = self.name`），
    #   与 `--robot.type` 的选择器**是两件事** —— lerobot 自己就是这么分的：
    #   `config_so_follower.py` 把同一个配置类注册成 `so101_follower` 与 `so100_follower`
    #   两个选择器，而 `SOFollower.name = "so_follower"`，两个选择器共用一个 `robot_type`。
    #
    #   所以这里取真机那个值：
    #     · 调用时怎么区分 → `--robot.type=so101_sim`（选择器，一直是分开的）
    #     · 数据里怎么记   → `robot_type=so_follower`，与真机逐字相同
    #
    #   为什么必须相同：官方合并的第一步 `validate_all_metadata`（aggregate.py:73）
    #   **逐字比 robot_type**，不等就 `raise ValueError` —— 仿真与真机就合不成一份
    #   训练集，而 `lerobot-train` 又只吃单一 repo_id。
    #   数据是仿真产的这件事由别处标着：目录名 `so101_sim_regen`、`repo_id`、
    #   任务描述、发布仓名 —— 都不必占用 `robot_type` 这个要参与合并校验的字段。
    name = "so_follower"

    def __init__(self, config: SO101SimRobotConfig):
        super().__init__(config)
        self.config = config
        self._env: So101SimEnv | None = None
        self._obs: dict[str, Any] | None = None
        self._frames: list[np.ndarray] = []
        self._states: list[np.ndarray] = []
        self._success: list[bool] = []
        # 逐项判据，键名就是环境 `evaluate()` 给的那些（`is_item_above_bin` /
        # `is_robot_static` / `robot_touching_item` / `robot_touching_bin` …）。
        self._criteria: dict[str, list[float]] = {}

    @property
    def observation_features(self) -> dict:
        """六个关节位置 + 每路相机的画面形状。

        ★形状取自**声明的常量**（`envs.py` 的 `CAMERA_NAMES` / `CAMERA_WIDTH` /
          `CAMERA_HEIGHT`），不从运行时环境读 —— `lerobot-record` 在 `connect()`
          **之前**就用这个属性算数据集的 features（`lerobot_record.py:502-512`），
          那时环境还没建。依赖运行时环境的写法会让录出来的数据集**没有图像特征**
          （实测：只剩 action / observation.state 与索引列，两路相机整个丢掉，
          全程不报错）。真机那侧 `so_follower._cameras_ft` 同样是从配置取形状。
        """
        shape = (CAMERA_HEIGHT, CAMERA_WIDTH, 3)
        return {**{f"{name}.pos": float for name in JOINT_NAMES},
                **{cam: shape for cam in CAMERA_NAMES}}

    @property
    def action_features(self) -> dict:
        """六个关节的绝对位置目标，真机口径。"""
        return {f"{name}.pos": float for name in JOINT_NAMES}

    @property
    def cameras(self) -> dict:
        """这台机器人有哪几路相机 —— 键是相机名，值是它的画面形状。

        `lerobot-record` 要用它：`num_image_writer_threads_per_camera * len(robot.cameras)`
        决定写图线程数。真机 follower 那侧 `cameras` 是真的相机对象字典；仿真这侧
        画面由 `get_observation` 一并给出，没有独立的相机对象，所以给形状就够。
        （`unitree_g1` 也是这么把 `cameras` 实现成属性的。）

        ★不能不实现：`lerobot_record.py:522` 用 `hasattr` 兜了底，但 548 行直接
        `len(robot.cameras)` 没兜 —— 少了它就是 `AttributeError`，而不是退化成 0。
        """
        shape = (CAMERA_HEIGHT, CAMERA_WIDTH, 3)
        return {cam: shape for cam in CAMERA_NAMES}

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
            # 集的边界由 lerobot 的命令管，环境不许自己 reset —— 见 So101SimEnv 的说明。
            auto_reset=False,
            sim_backend=self.config.sim_backend,
        )
        self._obs, _ = self._env.reset(seed=self.config.seed)
        if self.config.initial_state_path:
            self._obs = self._apply_initial_state(Path(self.config.initial_state_path))
        self._frames = []
        self._states = []
        self._success = []
        self._criteria = {}

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
        # ★ **夹爪这一路报真实关节位置，不要做成指令的回显。** 这条被改错过一次，
        #   代价是整轮数据作废，理由记在这里：
        #   当时的证据是「真机搬运段 state−action 只有 +0.64 点 ⇒ 它不携带爪口宽度」，
        #   而那个「搬运段」是用「集内最长的低指令连续段」选的 —— 300 集里它的起点分位
        #   中位是 **0.000**，选中的是**集开头空爪停在 home 闭合位**那一段，不是搬运段。
        #   按集内 15%~98% 重新选段：真机搬运段 state−action = **+13.78（cube）/
        #   +10.71（can）**。更决定性的是同集内保持段（指令恒定 ≥30 帧且 <5%）的
        #   段尾落差呈**干净的双峰**：0~1 点 538 段（空爪）、8~20 点 107 段（夹着东西），
        #   **2~8 点之间一个样本都没有**。指令按住一整秒不动、读数稳停在指令上方 14 点 ——
        #   滤波回显产生不了这个，那就是堵转，state 报的是被物体撑开的那个开度。
        # 落到编码器栅格上：臂五关节按度、带该机的格点相位；夹爪按它占满行程的比例。
        # 两者都逐环境查 `REAL_CALIBRATION` —— 不同对标机的标定跨度与相位都不一样。
        # 落到编码器栅格上：臂五关节按度，夹爪按它自己的行程换算出的量子。
        # 量化本身是真机每台都有的物理事实（12 位编码器），所以两边都要有；
        # 而**格点的相位不必对齐**：`_normalize` 里 `mid = (min_+max_)/2`，
        # 端点和为奇数时 mid 落在半格上、为偶数时落在整格上 —— 那是每台机标定
        # 端点的奇偶，没有物理含义，实测三台真机的相位各不相同。仿真是又一台机，
        # 它的 mid 是整数，相位 0 就是它自己的合法标定。
        pos[:5] = np.round(pos[:5] / ENCODER_STEP_DEG) * ENCODER_STEP_DEG
        grip_step = gripper_step_pct()
        pos[5] = np.round(pos[5] / grip_step) * grip_step
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
            # ★逐项记，不只记总判据。成功是四条的合取（物体在箱口内 ∧ 臂静止 ∧
            #   臂不碰物体 ∧ 臂不碰料箱），只看总判据分不出"物体没放进去"与
            #   "放进去了但手臂还在微动"—— 前者是数据废了，后者只是收尾没停稳。
            #   ★按 dtype 判数值，不用 `np.isscalar`：`info` 里还有个字符串项 `task`，
            #   而 numpy 的字符串标量也满足 `np.isscalar` ⇒ 放它过去之后 `float()` 抛错，
            #   把整条回放打断在第 1 帧。
            for key, value in info.items():
                arr = np.asarray(value)
                if key == "is_success" or arr.dtype.kind not in "biuf":
                    continue
                self._criteria.setdefault(key, []).append(float(arr.reshape(-1)[0]))
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
                     success=np.asarray(self._success[:n], dtype=bool),
                     **{k: np.asarray(v[:n], dtype=np.float32)
                        for k, v in self._criteria.items()})
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
            KeyError: 场景里有会动的物体而状态里没有它 —— 它会留在随机的复位位置，
                于是场景不是录制时那个，回放结果没有可比性。
        """
        import json

        import torch

        inner = self._env._env.unwrapped
        registry = inner.scene.state_dict_registry
        raw = json.loads(path.read_text())

        for group, known in (("actors", registry.actors), ("articulations", registry.articulations)):
            recorded = set(raw.get(group, {}))
            # 静态体在物理上不可能动，`get_state_dict` 也不收它们 —— 缺了无所谓。
            # 会动的缺了就必须停：它会留在复位位置，而复位位置是随机的。
            missing_movable = sorted(
                n for n in set(known) - recorded
                if getattr(known[n], "px_body_type", "dynamic") != "static"
            )
            if missing_movable:
                raise KeyError(
                    f"状态里缺这些会动的{group}：{missing_movable} —— 它们会留在复位位置，"
                    "而复位位置是随机的，场景就不是录制时那个了"
                )
            # 状态里多出来的是录制之后从场景里删掉的东西（例如只用于示意的标记物）。
            # 丢掉它们，但要报出来 —— 悄悄丢会让人以为状态是完整恢复的。
            extra = sorted(recorded - set(known))
            if extra:
                print(f"  置位时丢掉已不存在的{group}：{extra}")
                raw[group] = {k: v for k, v in raw[group].items() if k in known}

        def to_tensor(node):
            if isinstance(node, dict):
                return {k: to_tensor(v) for k, v in node.items()}
            return torch.as_tensor(np.asarray(node, dtype=np.float32), device=inner.device)

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
        return np.asarray(pixels[tops[0] if tops else min(pixels)])

    def _write_video(self) -> None:
        """把攒下的帧按场景帧率写成 mp4。"""
        import imageio.v2 as imageio

        path = Path(self.config.video_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fps = So101SimEnv.metadata["render_fps"]
        imageio.mimwrite(path, self._frames, fps=fps, macro_block_size=1)
