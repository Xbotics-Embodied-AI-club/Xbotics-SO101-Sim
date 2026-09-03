"""SO101KitSlow —— 把每步关节增量上限压到**真机实测速度包线**内的 KIT 机器人。

为什么要它：squint 默认控制器的每步上限在 20 fps 下折算成臂 114.6°/s、夹爪 229°/s，
而 reward 里没有任何速度代价 ⇒ 策略学成一路顶着上限跑，比真机实测的 p95 快 1.6–3.6 倍，
真机跟不上。修法不是改渲染帧率（那只是纸面变慢、单帧跳变照旧），而是把动作空间本身
压进真机包线再重训。逐关节反算而非一刀切一个系数 —— 各关节要压的倍数差两倍以上。

不改 vendored SO101，只在子类里覆盖控制器配置。逐关节对照表与系数取值的账见 bd xb-01ck。
"""

import copy

import numpy as np
from mani_skill.agents.controllers import deepcopy_dict
from mani_skill.agents.registration import register_agent

from so101_sim.robots.so101_base.so101 import SO101

# 控制频率 = 数据集帧率（一步一帧）。**对齐真机 task1 的 30fps**。
# ★注意 delta 上限是"每步"的量：帧率从 20 升到 30，若上限不变，
# 每秒步数多 1.5× ⇒ 角速度也涨 1.5×。所以下面反算时用的就是这个 SIM_FPS，
# 分母变大、上限自动缩小，速度不随帧率漂。
SIM_FPS = 30

# 在真机 p95 之上再整体放慢的系数，落到真机 p50–p95 之间。
# 按真机节奏本该取 0.2（真机取放段约 7.8 s，0.6 时策略只用 2.7 s），但 0.2 实测训不出来：
# 每步位移只剩 1/3、episode 预算涨 2.25 倍，末帧成功恒 0。见 bd xb-01ck。
SPEED_SCALE = 0.6

# 真机实测 p95 角速度，单位是**度每秒**（真机数据的 feature 名是 `shoulder_pan.pos`、
# 范围 −102…96，就是度），而仿真是弧度 —— 两边混算会虚高 57.3 倍。
REAL_P95_DEG_PER_S = {
    "shoulder_pan": 29.0,
    "shoulder_lift": 63.3,
    "elbow_flex": 65.9,
    "wrist_flex": 41.5,
    "wrist_roll": 31.6,
    "gripper": 84.4,
}


def per_step_limits(joint_names):
    """按真机 p95 × SPEED_SCALE 反算每步增量上限（rad/step），顺序与 `joint_names` 一致。"""
    return [float(np.deg2rad(REAL_P95_DEG_PER_S[name] * SPEED_SCALE / SIM_FPS))
            for name in joint_names]


class SlowControllerMixin:
    """把 `pd_joint_target_delta_pos` / `pd_joint_delta_pos` 的上下限换成真机包线值。

    只动这两个 delta 控制器（训练与数据生产用的就是 `pd_joint_target_delta_pos`）；
    `pd_joint_pos` / `pd_joint_vel` 原样保留，便于对照与回放。
    """

    @property
    def _controller_configs(self):
        """父类的控制器配置，其中两个 delta 控制器的上下限换成真机包线值。

        Returns:
            控制器配置字典的深拷贝 —— ManiSkill 会缓存并跨实例复用它，
            原地改会污染同一进程里别的环境。
        """
        configs = super()._controller_configs
        names = [j.name for j in self.robot.active_joints]
        limits = per_step_limits(names)
        for key in ("pd_joint_delta_pos", "pd_joint_target_delta_pos"):
            if key not in configs:
                continue
            cfg = copy.deepcopy(configs[key])
            cfg.lower = [-v for v in limits]
            cfg.upper = list(limits)
            configs[key] = cfg
        return deepcopy_dict(configs)


# 「夹得更稳」靠**物体摩擦**，不靠力门槛。
# squint 默认 `item_friction_range=(0.1,0.5)` 对橡胶指垫配这类件偏低太多，靠摩擦不够就只能
# 靠几何卡住，策略学不到捏紧 ⇒ 提到 0.8–1.0。
# ⚠️ 这是**物理假设、非实测** —— 有真机件材质配对的实测值就换成它。
#
# ★写死在这里，不从环境变量取。它是物理参数，**换个值就是另一份数据**；走环境变量时
#   "这批数据是按哪套摩擦产的"只存在于当时那条命令里，而同一份代码在两台机器上会产出
#   不同的数据，且两边都不报错。
ITEM_FRICTION_RANGE = (0.8, 1.0)

# 抓取力门槛沿用 `SO101.is_grasping` 的默认 0.5 N，本模块不覆盖它。
# 这里曾有一个 `FirmGraspMixin`，声称"把门槛提高、使轻贴不再算抓住"，而它传下去的
# min_force 与父类默认值逐字相同 —— 一个什么都没改的 mixin，却让读的人以为判据已经从严。
# 真要改门槛就改 `so101.py` 里那个默认值，只有那一处。


@register_agent()
class SO101KitSlow(SlowControllerMixin, SO101):
    """SO-101，动作空间压在真机速度包线内。"""

    uid = "so101_kit_slow"
