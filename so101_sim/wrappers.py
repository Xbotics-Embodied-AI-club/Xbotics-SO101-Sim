"""RL 训练用的观测包装与便利构造器。

这些包装不是格式转换，是 sim-to-real 的实质手段：

- **降采样到 16px**：低分辨率反而让视觉策略训得快、迁移得好（源自 squint 的做法）。
- **颜色抖动**：让策略不去死记仿真的配色，抗真实相机的色差。
- **``ignore_terminations=True``**：RL 要按固定步长稳定收集，不要一成功就断——
  这和评测时"成功即结束"的语义正好相反，所以两个入口不能共用同一套包装。

`visual_rl_env` 返回的是 ManiSkill 的标准 `ManiSkillVectorEnv`，不是自定义类型：
用的人拿到的还是原生生态里的对象，`reset`/`step` 就是 ManiSkill 的语义。
"""

from __future__ import annotations

import numpy as np  # noqa: F401  C 扩展 ABI 顺序：numpy 必须在 torch 之前
import torch
import torch.nn.functional as F
import torchvision
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import gymnasium as gym

from so101_sim._core import _make_maniskill

# `DownsampleObsWrapper` 与 `ColorJitterWrapper` 源自 squint（MIT），已转为本项目自维护。
# 与上游的差异记在 `so101_sim/tasks/UPSTREAM.md`。


class DownsampleObsWrapper(gym.ObservationWrapper):
    """把 RGB 观测从 render_size 降采样到 target_size（area 插值）。

    输入约定为 (B, H, W, C) 格式。
    """

    def __init__(self, env, target_size):
        super().__init__(env)
        self.target_size = target_size
        old_rgb_space = self.observation_space['rgb']
        C = old_rgb_space.shape[-1]
        self.observation_space['rgb'] = gym.spaces.Box(
            low=0, high=255, shape=(target_size, target_size, C), dtype=old_rgb_space.dtype
        )

    def observation(self, obs):
        """把 `obs['rgb']` 降采样到 `target_size`，其余键原样带过。

        Args:
            obs: 上游观测字典，`rgb` 形状 (B, H, W, C) 或 (H, W, C)。

        Returns:
            同一个字典（就地改 `rgb`）。已经是目标尺寸时直接返回，不做无谓的插值。
        """
        rgb = obs['rgb']
        if rgb.shape[-2] == self.target_size:
            return obs

        # ManiSkill 的原生观测有 batch 维、被 `squeeze` 过的没有，两种都要能进。
        squeeze = rgb.dim() == 3
        if squeeze:
            rgb = rgb.unsqueeze(0)

        # interpolate 要通道在前。area 插值而非 bilinear：降到 16px 时它更接近
        # 相机自身的像素平均，squint 的迁移结果就是在这个插值下取得的。
        rgb = rgb.permute(0, 3, 1, 2)
        rgb = F.interpolate(rgb.float(), size=(self.target_size, self.target_size), mode='area').to(torch.uint8)
        rgb = rgb.permute(0, 2, 3, 1)

        if squeeze:
            rgb = rgb.squeeze(0)

        obs['rgb'] = rgb
        return obs


class ColorJitterWrapper(gym.ObservationWrapper):
    """对 RGB 观测做随机颜色抖动，增强 sim2real 鲁棒性。

    输入约定为 (B, H, W, C)，通道数须为 3 的整数倍 —— 每 3 通道对应一路相机
    （`FlattenRGBDObservationWrapper` 把多路相机的 RGB 沿通道维拼接后就是这种排布）。
    每路独立抖动，因为两个物理相机的成色偏差与曝光噪声互不相关。
    """

    def __init__(self, env, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05):
        super().__init__(env)
        self.jitter = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    def observation(self, obs):
        """对每路相机分别做一次颜色抖动。

        Args:
            obs: 上游观测字典，`rgb` 是 uint8，通道数须为 3 的整数倍（每 3 通道一路相机）。

        Returns:
            同一个字典（就地改 `rgb`）。

        Raises:
            ValueError: 通道数不是 3 的整数倍 —— 那说明上游不是按「每路相机 3 通道」拼的，
                继续抖动会把相邻两路的通道混在一组里。
        """
        rgb = obs['rgb']

        # ManiSkill 的原生观测有 batch 维、被 `squeeze` 过的没有，两种都要能进。
        squeeze = rgb.dim() == 3
        if squeeze:
            rgb = rgb.unsqueeze(0)

        # torchvision 的 ColorJitter 只吃通道在前、值域 [0,1] 的浮点张量。
        rgb = rgb.permute(0, 3, 1, 2)
        rgb = rgb.float() / 255.0

        num_channels = rgb.shape[1]
        if num_channels % 3 != 0:
            raise ValueError(
                f"ColorJitterWrapper 只接受通道数是 3 的整数倍的输入（每 3 通道对应一路"
                f"相机），但收到了 {num_channels} 通道。"
            )
        # 逐组调用而非采样一次复用：每次调用重新抽亮度/对比度/饱和度/色调，
        # 于是两路相机拿到互不相关的色差，与两个物理传感器的实际情形一致。
        groups = [self.jitter(rgb[:, i:i + 3]) for i in range(0, num_channels, 3)]
        rgb = torch.cat(groups, dim=1)

        rgb = rgb.permute(0, 2, 3, 1)
        rgb = (rgb.clamp(0, 1) * 255).to(torch.uint8)

        if squeeze:
            rgb = rgb.squeeze(0)

        obs['rgb'] = rgb
        return obs


def visual_rl_env(task: str, num_envs: int, image_size: int = 16, render_size: int = 128,
                   domain_randomization: bool = True) -> ManiSkillVectorEnv:
    """给视觉 RL 用的批量环境：原生环境 + 展平 + 降采样 + 颜色抖动 + 向量化。

    Args:
        task: 已注册的环境 id。
        num_envs: 批量份数。
        image_size: 送进策略的方形边长。
        render_size: 渲染边长，大于 `image_size` 时先渲后降采样。
        domain_randomization: 是否开启域随机化。

    Returns:
        `ManiSkillVectorEnv`，`ignore_terminations=True`（RL 要固定步长收集）。
    """
    # 这里宽高同值是故意的：RL 观测是方形小图，不重建真机画面。
    # 要与真机同构的画面走 `So101SimEnv`，它默认不覆盖标定分辨率。
    raw = _make_maniskill(task, num_envs=num_envs, obs_mode="rgb+segmentation",
                          sensor_width=render_size, sensor_height=render_size,
                          render_mode="all",
                          domain_randomization=domain_randomization)
    env = FlattenRGBDObservationWrapper(raw, rgb=True, depth=False, state=True)
    if render_size != image_size:
        env = DownsampleObsWrapper(env, target_size=image_size)
    env = ColorJitterWrapper(env)
    return ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)


def state_rl_env(task: str, num_envs: int, domain_randomization: bool = True) -> ManiSkillVectorEnv:
    """只给关节状态向量的批量环境（跳过渲染管线），供状态策略与调试用。

    Args:
        task: 已注册的环境 id。
        num_envs: 批量份数。
        domain_randomization: 是否开启域随机化。

    Returns:
        `ManiSkillVectorEnv`，观测是状态向量，不含画面。
    """
    # `obs_mode="state"` 不渲染，相机尺寸对观测没有影响，给个占位值即可。
    raw = _make_maniskill(task, num_envs=num_envs, obs_mode="state",
                          sensor_width=128, sensor_height=128,
                          render_mode="all", domain_randomization=domain_randomization)
    return ManiSkillVectorEnv(raw, num_envs, ignore_terminations=True, record_metrics=True)
