"""共享的 ManiSkill 批量环境构造核。

`lerobot_env.So101SimEnv`（lerobot 单环境评测口）和 `wrappers.visual_rl_env`（RL 批量训练口）
都从这一句 `gym.make` 出发，只是 num_envs / obs_mode / 渲染参数不同——保证两条消费路径拿到的
是同一个底层仿真定义（同一批任务、同一份机器人资产），不会各自漂移出两套环境。
"""

from __future__ import annotations

import gymnasium as gym


def _make_maniskill(
    task: str,
    num_envs: int,
    obs_mode: str,
    sensor_width: int | None,
    sensor_height: int | None,
    render_mode: str,
    domain_randomization: bool = False,
    control_mode: str | None = None,
    max_episode_steps: int | None = None,
):
    """构造一个原始（未包装）的批量 ManiSkill 环境，首维恒为 num_envs。

    `control_mode` 决定动作向量的语义，**必须与数据集里 `action` 的口径一致**：
    - 默认（None）用机器人自己的默认模式 `pd_joint_target_delta_pos`：动作是
      **归一化增量**，在控制器内部目标上累加。
    - `"pd_joint_pos"`：动作是**绝对关节角目标**。已交付数据集与本仓库的脚本化产线
      录的都是这个口径。

    ★两者混用不会报错，只会安静地跑错：把绝对角（约 −2.0~1.6 rad）当归一化增量喂进去，
      每一维都会被 clip 到 ±1，手臂以包线最大速度朝错误方向走。评测一个在绝对角数据上
      训出来的策略时若忘了指定这个参数，得到的低成功率会被误读成"策略没学会"。

    ★另一处同类陷阱是 `max_episode_steps`：三个分发任务注册的是 400 步，够装下脚本化
      产线的轨迹（中位 368 帧），但**装不下更长的轨迹**。拿一条 ~550 帧的数据训出来的
      策略，在 400 步的环境里会在完成动作之前被截断 —— 同样不报错，同样得到一个会被
      误读成"策略没学会"的低成功率。数据集轨迹比 400 长时，这里显式放宽。

    Args:
        task: 已注册的环境 id。
        num_envs: 批量份数，观测与奖励的首维。
        obs_mode: ManiSkill 观测模式，如 `"rgb"` / `"state"`。
        sensor_width: 相机宽，`None` 表示不覆盖环境自己的标定分辨率。
        sensor_height: 相机高，与 `sensor_width` 同时给或同时不给。
        render_mode: ManiSkill 渲染模式，`"all"` 才含第三人称画面。
        domain_randomization: 是否开启域随机化。
        control_mode: 见上文；`None` 用机器人默认模式。
        max_episode_steps: 覆盖任务注册的单集步数上限，`None` 表示不覆盖。

    Returns:
        未包装的批量 ManiSkill 环境。

    Raises:
        ValueError: `sensor_width` 与 `sensor_height` 只给了一边。相机的竖直视野角是
            ChArUco 标定值（fovy 59.17°）且固定，水平视野由宽高比推出来 ⇒ 只改一边
            等于换了一台相机，渲出来的画面与真机、与已交付数据集不再同构。
    """
    kwargs = {}
    if control_mode is not None:
        kwargs["control_mode"] = control_mode
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    if (sensor_width is None) != (sensor_height is None):
        raise ValueError(
            "sensor_width 与 sensor_height 必须同时给或同时不给 —— "
            "只改一边会在固定的竖直视野角下改掉宽高比，水平视野随之改变。"
        )
    # 都不给就不下发 sensor_configs，让环境 CameraConfig 里的标定分辨率生效 ——
    # 数据产线走的正是这条路，于是评测画面默认与训练数据同构。
    if sensor_width is not None:
        kwargs["sensor_configs"] = dict(width=sensor_width, height=sensor_height)
    return gym.make(
        task,
        num_envs=num_envs,
        obs_mode=obs_mode,
        sim_backend="gpu",
        render_mode=render_mode,
        domain_randomization=domain_randomization,
        **kwargs,
    )
