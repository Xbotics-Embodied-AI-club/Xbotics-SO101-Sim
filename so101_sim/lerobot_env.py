"""把 ManiSkill3 SO101 任务包装成 lerobot 评测用的单环境 gym.Env。

lerobot 的 `make_env` 对 `so101_sim` 走通用分支：先 `import so101_sim` 触发注册
（见 `__init__.py`），再用 `SyncVectorEnv` 把若干份本环境包成向量环境。ManiSkill 天生
是批量张量（首维 = num_envs），这里默认 num_envs=1、对首维取 `[0]`，对上层呈现成普通单
环境接口——和 lerobot 自带的 `LiberoEnv` 一样，评测代码无需知道底层是 ManiSkill。

`num_envs` 参数保留了泛化到批量的能力：obs 格式化、动作 reshape、reward 与 success 都按
batch 维处理，`num_envs=1` 时对首维取 `[0]`，于是 lerobot 的评测契约仍然成立。

这是本模块与仿真器之间**唯一的耦合点**：只有本文件依赖 `so101_sim.tasks` / ManiSkill，
其它一切（评测、rollout、换色、机器人伪装）都只依赖 `So101SimEnv`。
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from so101_sim._core import _make_maniskill


def _to_numpy(x: Any) -> np.ndarray:
    """ManiSkill 返回 GPU torch 张量；统一搬到 CPU numpy。"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class So101SimEnv(gym.Env):
    """一个 SO-101 抓放场景的单环境视图，观测按 lerobot 评测约定给出。

    Attributes:
        task: 底层 ManiSkill 环境 id。
        obs_type: `"pixels"` 或 `"pixels_agent_pos"`。
        num_envs: 底层批量份数；为 1 时对外呈现成普通单环境。
        observation_width: 实际生效的相机宽，从底层相机配置读出。
        observation_height: 实际生效的相机高，从底层相机配置读出。
        unit_convention: 状态与动作对外的口径，`"real"`（真机口径）或 `"maniskill"`（原生弧度）。
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task: str = "SO101PickPlaceCube40-v1",
        obs_type: str = "pixels_agent_pos",
        obs_mode: str = "rgb",
        render_mode: str = "rgb_array",
        observation_width: int | None = None,
        observation_height: int | None = None,
        episode_length: int = 400,
        num_envs: int = 1,
        control_mode: str | None = None,
        unit_convention: str = "real",
        auto_reset: bool = True,
        sim_backend: str = "physx_cpu",
        sim_config: dict | None = None,
        **kwargs,
    ):
        """建一个已注册场景的单环境视图。

        有三个参数必须与「被评策略所训数据是怎么产生的」对齐，配错都**不报错、
        只会安静跑错**，而三者都表现为一个会被误读成「策略没学会」的低成功率：

        - `control_mode`：已交付数据集录的是绝对关节角，要传 `"pd_joint_pos"`。
          不传则用机器人默认的归一化增量模式，绝对角会被逐维 clip 到 ±1，
          手臂以包线最大速度朝错误方向走。
        - `episode_length`：要装得下数据集里的轨迹长度。三个分发任务注册的是 400 步，
          够装脚本化产线的轨迹（中位 368 帧），但装不下更长的。
        - `unit_convention`：ManiSkill 内部一律弧度，而真机（`lerobot-record` 走
          `so_follower`）的口径是**混的** —— 五个臂关节是度，**夹爪是 0~100 的行程
          百分比**（`so_follower` 把 gripper 写死为 `MotorNormMode.RANGE_0_100`，
          与 `use_degrees` 无关）。所以「统一到真机」不是一个单位换算，是逐通道换算。
          默认 `"real"`；`"maniskill"` 保持原生弧度，给直接对着 ManiSkill 写的调用方。

          配错的代价：臂关节差 57.3 倍 —— 观测偏小会让归一化输出远离训练分布，动作
          偏大会逐维顶到关节限位。夹爪那一维更隐蔽：度数与百分比的**量级恰好撞车**
          （物理行程约 0~100 度），所以看数值看不出来，只表现为抓取这一环学不动。

        Args:
            task: 已注册的环境 id（三个分发场景之一）。
            obs_type: `"pixels"` 只给画面，`"pixels_agent_pos"` 另给关节位置。
            obs_mode: 传给 ManiSkill 的观测模式。
            render_mode: 传给 ManiSkill 的渲染模式。
            observation_width: 相机宽。`None`（默认）表示不覆盖环境自己的标定分辨率 ——
                数据产线就是直接用标定值渲的，所以默认不覆盖时本入口的画面与已交付
                数据集同构，不依赖调用方记得填对数字。
            observation_height: 相机高，与 `observation_width` 同时给或同时不给。
            episode_length: 单集步数上限，同时下发给底层环境。只改本类属性而底层仍是
                注册值的话，超长行为会被 ManiSkill 的 TimeLimit 截断而评测端看不出来。
            num_envs: 底层批量份数。
            control_mode: 动作语义。`None` 用机器人默认模式。
            unit_convention: 状态与动作对外的口径。`"real"`（默认）= 真机口径：
                五个臂关节度制、夹爪 0~100 行程百分比；`"maniskill"` = 原生弧度。
                只影响 `agent_pos` 与动作，不影响画面。
            sim_backend: 物理后端，默认 `"physx_cpu"`。本类是**单环境**入口
                （lerobot 的 record / replay / eval 都走它），而 GPU PhysX 在
                `num_envs=1` 下每控制步要 47.96ms（20.9 Hz）、跑不到 30 Hz，
                CPU PhysX 是 3.24ms（308 Hz）。子步数不受影响，见 `_make_maniskill`。
            auto_reset: 本集结束时是否就地 reset。`True`（默认）满足 gym 契约，
                `lerobot-eval` 靠它连续跑多集。**驱动机器人插件时必须给 `False`**：
                集的边界由 `lerobot-record` / `lerobot-replay` 管，而 `terminated`
                里并了 `success` ⇒ 一成功环境就换场景、手臂弹回 home，调用方却还在
                按原轨迹发动作。实测表现是「回放到某帧后手臂暴走 81°、物体飞在空中」，
                极像物理不可复现，其实是自己把场景换了。
            sim_config: 透给 ManiSkill 的 `SimConfig` 覆盖（如
                `{"scene_config": {"contact_offset": 0.002}}`）。`None` 表示不覆盖。
            **kwargs: 容纳 lerobot 传来的其它环境参数，**一律忽略但会打印出来**。

                ★早先这里只写"忽略"、什么也不说，于是拼错的参数名与本类不认识的参数
                都被静默吞掉。实账：为了查"两后端接触行为为什么不同"，我用
                `sim_config=...` 扫了 `gpu_memory_config` 与 `contact_offset` 各五档，
                两轮结果**逐位相同** —— 当时以为"参数无效"，其实是参数从没送到环境。
                两轮扫描白做。所以现在忽略要**说出来**。

        Raises:
            ValueError: `observation_width` 与 `observation_height` 只给了一边（等于在
                标定好的竖直视野角下改宽高比，水平视野随之改变）；各路相机尺寸不一致；
                或 `unit_convention` 不是 `"real"` / `"maniskill"`。
            NotImplementedError: `obs_type` 不是支持的两个取值之一。
        """
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.num_envs = num_envs
        if unit_convention not in ("real", "maniskill"):
            raise ValueError(
                f"unit_convention 只能是 'real' 或 'maniskill'，收到 {unit_convention!r}"
            )
        self.unit_convention = unit_convention
        self.auto_reset = auto_reset
        if (observation_width is None) != (observation_height is None):
            raise ValueError(
                "observation_width 与 observation_height 必须同时给或同时不给："
                "只改一边等于在标定好的竖直视野角下改宽高比，水平视野会跟着变，"
                "渲出来的画面与真机、与已交付数据集不再同构。"
            )
        # lerobot 的 rollout 用 env.call("_max_episode_steps") 界定单集步数上限。
        self._max_episode_steps = episode_length
        if kwargs:
            print(f"So101SimEnv 忽略了这些参数：{sorted(kwargs)}")

        self._env = _make_maniskill(
            task,
            num_envs=num_envs,
            obs_mode=obs_mode,
            sensor_width=observation_width,
            sensor_height=observation_height,
            render_mode=render_mode,
            control_mode=control_mode,
            max_episode_steps=episode_length,
            sim_backend=sim_backend,
            sim_config=sim_config,
        )

        # 相机名与尺寸都从环境实际生效的配置读，不写死、也不从构造参数推：
        # 构造参数可能是 None（不覆盖），此时只有底层配置知道真实尺寸，
        # 而观测空间一旦与实际吐出的形状分岔就没有任何一步会报错。
        sensor_configs = self._env.unwrapped._sensor_configs
        self._camera_names = sorted(sensor_configs)

        first = sensor_configs[self._camera_names[0]]
        self.observation_height = int(first.height)
        self.observation_width = int(first.width)
        for name in self._camera_names[1:]:
            config = sensor_configs[name]
            if (int(config.height), int(config.width)) != (self.observation_height, self.observation_width):
                raise ValueError(
                    f"相机 {name} 是 {config.width}×{config.height}，"
                    f"与 {self._camera_names[0]} 的 {self.observation_width}×{self.observation_height} 不同 —— "
                    "本入口按「每路相机同尺寸」给出观测空间，不同尺寸时请显式处理"
                )

        image_space = spaces.Box(
            low=0, high=255,
            shape=(self.observation_height, self.observation_width, 3), dtype=np.uint8,
        )
        single_act = getattr(self._env, "single_action_space", self._env.action_space)
        self._action_dim = int(np.prod(single_act.shape[-1:]))
        pixels_space = spaces.Dict({name: image_space for name in self._camera_names})
        if obs_type == "pixels":
            self.observation_space = spaces.Dict({"pixels": pixels_space})
        elif obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": pixels_space,
                    "agent_pos": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(self._action_dim,), dtype=np.float32
                    ),
                }
            )
        else:
            raise NotImplementedError(
                f"obs_type '{obs_type}' 不支持；用 'pixels' 或 'pixels_agent_pos'。"
            )

        # 夹爪的百分比映射用**底层关节自己的限位**算，不写死数字 ——
        # URDF 换了限位，这里必须跟着换，而写死的常数不会跟着换也不会报错。
        gripper = self._env.unwrapped.agent.robot.active_joints[self._gripper_index]
        if "gripper" not in gripper.name:
            raise ValueError(
                f"最后一个关节是 {gripper.name!r}，不是夹爪 —— 夹爪的百分比映射按「最后一维」"
                "取，关节顺序变了必须当场停下，否则会把某个臂关节当成夹爪换算。"
            )
        limits = np.asarray(gripper.limits.cpu()).reshape(-1)[:2]
        self._gripper_rad_lo, self._gripper_rad_hi = float(limits[0]), float(limits[1])

        # 动作空间的界也要跟着口径走，否则调用方按界裁剪出来的动作是另一套刻度。
        self.action_space = spaces.Box(
            low=self._from_sim(np.asarray(single_act.low, dtype=np.float32).reshape(-1)),
            high=self._from_sim(np.asarray(single_act.high, dtype=np.float32).reshape(-1)),
            shape=(self._action_dim,),
            dtype=np.float32,
        )

    # 真机口径下夹爪是行程百分比而不是角度 —— `so_follower` 把 gripper 写死为
    # `MotorNormMode.RANGE_0_100`，与 `use_degrees` 无关。夹爪恒为最后一维。
    _gripper_index = -1

    def _from_sim(self, x: np.ndarray) -> np.ndarray:
        """把 ManiSkill 的弧度换成对外口径（臂关节→度，夹爪→行程百分比）。"""
        if self.unit_convention != "real":
            return x
        out = np.rad2deg(np.asarray(x, dtype=np.float32))
        span = self._gripper_rad_hi - self._gripper_rad_lo
        pct = (np.asarray(x, dtype=np.float32)[..., self._gripper_index]
               - self._gripper_rad_lo) / span * 100.0
        out[..., self._gripper_index] = pct
        return out.astype(np.float32)

    def _to_sim(self, x: np.ndarray) -> np.ndarray:
        """把对外口径换回 ManiSkill 的弧度（`_from_sim` 的逆）。"""
        if self.unit_convention != "real":
            return x
        out = np.deg2rad(np.asarray(x, dtype=np.float32))
        span = self._gripper_rad_hi - self._gripper_rad_lo
        rad = (np.asarray(x, dtype=np.float32)[..., self._gripper_index] / 100.0 * span
               + self._gripper_rad_lo)
        out[..., self._gripper_index] = rad
        return out.astype(np.float32)

    def _format_raw_obs(self, raw_obs: dict) -> dict:
        """把 ManiSkill 的批量张量观测转成 lerobot 约定的 numpy 字典。

        ManiSkill 那侧的键是 `sensor_data.<相机名>.rgb`（N,H,W,3）与
        `agent.noisy_qpos`（N,dof）。键名沿用上游，但**这条路径上它就是干净 qpos** ——
        噪声只在 `domain_randomization` 为真时加（`place.py` 的 `_get_obs_agent`），
        而本入口从不打开它。要评噪声鲁棒性得另开域随机化，不能靠这个键名。

        Args:
            raw_obs: ManiSkill 的原始观测字典。

        Returns:
            `{"pixels": {相机名: 图}}`，`obs_type` 含关节位置时另有 `"agent_pos"`。
            `num_envs=1` 时对首维取 `[0]`，否则整批返回。
        """
        images = {}
        for name in self._camera_names:
            rgb = _to_numpy(raw_obs["sensor_data"][name]["rgb"]).astype(np.uint8)
            images[name] = rgb[0] if self.num_envs == 1 else rgb
        if self.obs_type == "pixels":
            return {"pixels": images}
        agent_pos = self._from_sim(_to_numpy(raw_obs["agent"]["noisy_qpos"]).astype(np.float32))
        agent_pos = agent_pos[0] if self.num_envs == 1 else agent_pos
        return {"pixels": images, "agent_pos": agent_pos}

    def reset(self, seed: int | None = None, **kwargs) -> tuple[dict, dict]:
        """重置到该场景的开机位姿。

        Args:
            seed: 随机种子，决定物体的生成位置。
            **kwargs: 忽略，容纳 gym 调用方传来的其它参数。

        Returns:
            `(观测, info)`；`info["is_success"]` 恒为假，成功只可能在 `step` 之后出现。
        """
        super().reset(seed=seed)
        raw_obs, _ = self._env.reset(seed=seed)
        info = {"is_success": False} if self.num_envs == 1 else {"is_success": np.zeros(self.num_envs, dtype=bool)}
        return self._format_raw_obs(raw_obs), info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict[str, Any]]:
        """走一步。动作的语义由 `control_mode` 决定，口径由 `unit_convention` 决定。

        Args:
            action: 形状 `(dof,)`，`num_envs != 1` 时为 `(num_envs, dof)`；
                口径与 `unit_convention` 一致（默认真机口径：臂关节度、夹爪百分比）。

        Returns:
            `(观测, reward, terminated, truncated, info)`。成功即计入 `terminated`；
            单环境且本集结束时另给 `info["final_info"]`（与 lerobot 的 `LiberoEnv` 一致），
            并就地 reset 让下一集从头开始。
        """
        act = self._to_sim(np.asarray(action, dtype=np.float32).reshape(self.num_envs, self._action_dim))
        raw_obs, reward, terminated, truncated, info = self._env.step(act)

        success = _to_numpy(info["success"]).reshape(self.num_envs).astype(bool)
        reward_batch = _to_numpy(reward).reshape(self.num_envs).astype(np.float32)
        terminated_batch = _to_numpy(terminated).reshape(self.num_envs).astype(bool) | success
        truncated_batch = _to_numpy(truncated).reshape(self.num_envs).astype(bool)
        observation = self._format_raw_obs(raw_obs)

        if self.num_envs != 1:
            out_info: dict[str, Any] = {"task": self.task, "is_success": success, "done": terminated_batch}
            return observation, reward_batch, terminated_batch, truncated_batch, out_info

        is_success = bool(success[0])
        reward_out = float(reward_batch[0])
        terminated_out = bool(terminated_batch[0])
        truncated_out = bool(truncated_batch[0])
        out_info = {"task": self.task, "is_success": is_success, "done": terminated_out}
        if terminated_out or truncated_out:
            out_info["final_info"] = {"task": self.task, "is_success": is_success, "done": True}
            # ★只有 gym 契约那一侧需要就地 reset（`lerobot-eval` 靠它连续跑多集）。
            #   机器人插件那一侧**必须不 reset**：集的边界由 `lerobot-record` /
            #   `lerobot-replay` 管，环境自己重置会把场景重新撒点、手臂弹回 home，
            #   而调用方还在按原轨迹发动作。实测表现是「回放到某一帧后手臂暴走 81°、
            #   物体飞在空中」—— 看起来像物理不可复现，其实是自己把场景换了。
            #   触发条件很隐蔽：`terminated` 里并了 `success`，所以**一成功就换场景**。
            if self.auto_reset:
                self.reset()
        return observation, reward_out, terminated_out, truncated_out, out_info

    def render(self) -> np.ndarray:
        """给一帧画面。`render_mode="all"` 时是三路横向拼接（含第三人称）。

        Returns:
            `(H, W, 3)` 的 uint8 数组。
        """
        frame = _to_numpy(self._env.render())
        return frame[0].astype(np.uint8) if frame.ndim == 4 else frame.astype(np.uint8)

    def close(self):
        self._env.close()
