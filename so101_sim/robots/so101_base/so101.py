"""SO-101 机器人 agent，加载真机套件几何 `kit_assets/kit_v1_so101.urdf`。

本包只有这一份几何，`wrist_roll` 的零位按实物安装标定（该关节真机不做标定）。
几何里除 6 个活动关节外还有套件底板、型材框，以及两个相机的安装座与光学系
（`top_camera_optical_frame` / `wrist_camera_optical_frame`，都是 fixed 关节、不加自由度）。
相机 sensor 不在这里定义，而在 `so101_sim.envs` 里用 CameraConfig 绑到那两个光学系上。

夹爪行程与关节限位一律从这份 URDF 现读 —— 换算标度取错会静默改变夹爪百分比。
"""

import copy
import xml.etree.ElementTree as ET

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from pathlib import Path

_URDF_PATH = str(Path(__file__).parent.parent / "kit_assets" / "kit_v1_so101.urdf")


def gripper_limit_rad():
    """夹爪关节的限位（弧度），从 URDF 现读。

    Returns:
        `(下限, 上限)`。

    Raises:
        ValueError: URDF 里没有名为 `gripper` 的关节或它没有限位。

    夹爪的 0~100 行程百分比就是按这个区间定义的（`lerobot_env` 里换算观测时同样从
    这里现读），所以两处不许各抄一份数 —— 换 URDF 时只有这一处要跟着变。
    """
    for joint in ET.parse(_URDF_PATH).getroot().iter("joint"):
        if joint.get("name") == "gripper":
            limit = joint.find("limit")
            if limit is not None:
                return float(limit.get("lower")), float(limit.get("upper"))
    raise ValueError(f"{_URDF_PATH} 里读不到 gripper 关节的限位")


def grip_pct_from_rad(rad):
    """夹爪关节角（弧度）→ 行程百分比。

    ★ **两边同一把尺子，不要在这里加标度。** 真机 leader 的 0~100 跨 1260 个计数
      = 110.74°（最小二乘从 300 集 `action[5]` 的格点反解，776 个取值 100% 落在格上），
      URDF 是 110.00° —— 差 0.7%，而真机两台机自己就差 1.5%（另一台 1279 计数）。
    ★ 曾按「空爪 1.896 / 夹 40mm 16.012」两点仿射把标度改成 0.494，被交叉验证否掉：
      高尔夫球是刚体标准件 42.67mm，那把尺子把它预测成 53.7mm（偏 26%）。
      cube 与 golf 的残差**同号** ⇒ 那一段落差是**堵转偏置**（舵机读数系统性低于真实
      爪口宽度）加上真机方块是海绵会压扁，属物性差异，只能平移修，改斜率会把锚点
      之外的整条曲线拧歪（仿真张开端从真机 p97 掉到 p72）。
    """
    lo, hi = gripper_limit_rad()
    return (rad - lo) / (hi - lo) * 100.0


def grip_rad_from_pct(pct):
    """行程百分比 → 夹爪关节角（弧度）。

    低于 0 的指令会解出小于关节下限的目标位 —— 那正是"捏到底"：
    目标压在物理止点之外，靠物体把两指卡死。
    """
    lo, hi = gripper_limit_rad()
    return lo + pct / 100.0 * (hi - lo)


# 真机开机位姿。判据是**落在真机各源的 home 区间内且靠近中位**，不是"与某一份逐维吻合"
# —— modelscope 那 9 份真机数据集的 home 彼此就差很多（实测各源 ep0 停在 home 那几帧：
# shoulder_pan 跨度 10.7°、wrist_flex 21.6°、elbow 6.0°、wrist_roll 6.4°），
# 各台舵机零位标定各自不同，不存在一个能同时吻合所有源的 home。本值逐维都在区间内、
# 靠中位，这也正是"仿真可以当作又一台真机"的依据。
# 换成别的起始位形，策略从没见过的位姿起步，实测同一份权重成功率 0/20 → 9/10。
#
# ★ 臂关节与夹爪**不是同一个量纲**，所以分成两个常量：lerobot 的 `so_follower` 把
#   gripper 写死成 `MotorNormMode.RANGE_0_100`，只有臂五关节走度制 ⇒ 真机数据里夹爪
#   那一列是行程百分比。曾把六个值放在一个列表里一律 `np.radians()`，于是百分比被当成
#   角度：仿真 home 夹爪读 10.818 而真机读 1.896，两侧闭合端取值区间完全不重叠 ——
#   混训时同一个物理状态在两份数据里是两个不同的数。
REAL_HOME_ARM_DEG = [-5.76, -102.68, 92.97, 63.38, -0.53]
REAL_HOME_GRIPPER_PCT = 1.90

# 各关节的力矩上限（N·m）。臂五关节与夹爪**必须分开**。
#
# 真机遥操里人是把扳机**捏到底**的，抓紧靠物体把两指卡死 —— 真机夹爪 action 中位 1.98%、
# state 中位 14.35%，那 12.4 点落差就是堵转。所以仿真收到"合到底"时也必须堵转。
#
# 夹爪取 0.20。这个值管两件事，**第二件是后来才量明白的**：
#
# ① 收到"捏到底"时要堵转，不能把物体挤飞。实测扫描（`measure_squeeze.py`，抓着 40mm
#    刚体方块、悬空、下发 0%、走 40 步）——
#      上限 10（URDF 声明的 effort）→ 合到 0.00%，方块被挤出 115mm
#      上限  5 → 合到 0.00%，挤出 79mm
#      上限  2 → 停在 23.38%，方块只挪 13.3mm
#      上限  1 / 0.5 / 0.2 → 停在 24.4 / 26.8 / 27.3%，挪 15.5 / 16.3 / 18.0mm
#    ⇒ 2.0 及以下都能堵转。这一条**不能单独定值**。
#
# ② 堵转段爪子抖不抖。目审逐帧读出「动指那一块永远在抖」，量化之后是**频率**问题不是
#    幅度问题：仿真峰峰 5.72% 比真机（17.16%）还小，而方向翻转率 20.11 vs 0.18 次/秒。
#    机理：捏到底时目标位在物理止点之外约 0.4 rad，PD 力矩（stiffness 1e3 × 0.4）远超
#    本上限 ⇒ 驱动**饱和**，而 PhysX 把驱动的阻尼项算在同一个被截断的和里 ⇒ 速度反馈
#    失去权限。**每步注入的能量正比于本上限**，于是它直接决定这个极限环的强弱。
#    实测（`probe_jaw_chatter.py`，cube40 每点 40 集，堵转段方向翻转率 / 峰峰值 /
#    物体在夹爪坐标系里逐帧位移 / 落点成功率）——
#      2.00 → 20.19 次/秒　5.64%　0.565mm　19/20
#      0.75 →  3.30       3.06%  0.148mm  39/40
#      0.25 →  1.80       1.39%  0.066mm  39/40
#      0.20 →  1.25       0.55%  0.041mm  39/40　← 取它
#      0.15 →  2.40       0.43%  0.029mm  39/40
#    **成功率在 0.15~2.0 之间完全不变，而抖动差 16 倍、物体在爪内的位移差 14 倍。**
#
# ★ 取舍要讲明白：2.0 当初是照真机舵机量级取的（STS3215 堵转约 30 kg·cm ≈ 2.9 N·m）。
#   0.20 **离开了数据手册**，是为了修一个求解器假象。判据取「观感更像真机 + 成功率不掉」：
#   真机舵机堵转不嗡嗡响（0.18 次/秒），仿真在 2.0 下每秒来回二十次，在 0.20 下 1.25 次。
#   物理量对不上手册、而可观测行为对上了 —— 这条产线要的是后者。
#
# ★ 这个值决定仿真能不能当混训策略的评测环境：策略从真机数据学到的输出就是"捏到底"，
#   若仿真收到它就把物体挤飞，那策略在仿真里永远抓不住 —— 仿真失去评测价值。
#
# 臂五关节保持 100：它们要举起整条臂加负载，实测按这个值能复现真机轨迹（同一串动作
# 驱动，量回的关节角中位差 0.4~1.7°）。改小会让跟随变差，那是另一条账。
ARM_FORCE_LIMIT = 100.0
GRIPPER_FORCE_LIMIT = 0.20
JOINT_FORCE_LIMITS = [ARM_FORCE_LIMIT] * 5 + [GRIPPER_FORCE_LIMIT]

# 夹爪关节的库仑摩擦（N·m）。URDF 里这个关节**没有 `<dynamics>`**，摩擦与阻尼都是 0。
#
# 为什么必须补：捏到底时目标位在物理止点之外约 0.4 rad，PD 算出的力矩
# （stiffness 1e3 × 0.4 = 400 N·m）远超 `GRIPPER_FORCE_LIMIT`，于是驱动**饱和**——
# 而 PhysX 把驱动的阻尼项算在同一个被截断的和里面，速度反馈因此失去权限。爪子被一个恒定
# 2 N·m 顶着，接触求解器的任何回弹都没人压制。目审逐帧读出来的就是这个：合拢后保持的
# 12 帧里方块仍在转、仍在往爪腔里滑，右侧那块爪的台阶侧影逐帧变形（用户：「夹爪一直在抖」）。
#
# 关节摩擦是**独立于驱动的**库仑摩擦，不被 `force_limit` 截断，所以它能补上这个洞的一半 ——
# 真机伺服堵转不嗡嗡响，靠的也有减速箱这一路摩擦。
#
# ⚠️ **但它压不住那个极限环，这一点后来被实测证伪了。** 加上它之后目审仍然读出
#   「动指那一块永远在抖」，量化：堵转段方向翻转率对摩擦几乎不敏感 ——
#     摩擦 0.2 / 0.4 / 0.6 / 0.8 / 1.0 / 1.4 → 19.89 / 20.68 / 20.22 / 19.90 / 19.82 / 19.51 次/秒
#   把力矩上限降到 0.75 之后再扫一遍摩擦（0.2 / 0.4 / 0.6 → 3.47 / 4.00 / 3.96）同样是平的。
#   ⇒ 摩擦压住的是**幅度**，压不住**频率**；真正的能量源是 `GRIPPER_FORCE_LIMIT`（见上）。
#   保留 0.2 是因为它无害且确实减小幅度，**不要再把它当成抖动的解药**。
GRIPPER_JOINT_FRICTION = 0.2


def kit_rest_qpos():
    """起始位姿的弧度形式。

    Returns:
        六个关节角（弧度）：前五维由度换算，夹爪由**真机口径**的行程百分比换算
        （见 `grip_rad_from_pct`：那把尺子标定到真机，不是 URDF 区间）。
    """
    return np.radians(REAL_HOME_ARM_DEG).tolist() + [
        grip_rad_from_pct(REAL_HOME_GRIPPER_PCT)
    ]


# ── 关掉 4 对**假**自碰撞 ──────────────────────────────────────────────────
#
# URDF 的碰撞几何是 STL 的凸包，比真件胖。后果实测：命令机器人保持真机 home 位姿时，
# `shoulder_link↔lower_arm_link` 接触力 130N 起步、几步内飙到 **18000N**，手臂被弹飞
# （shoulder_pan −5.8°→+115°），shoulder_lift 被顶到关节限位 —— 于是仿真**回不到真机 home**。
#
# 哪几对是"假"的，用可证伪的判据定：真机 106085 帧里出现过的关节角，物理上一定不自碰
# （真臂就那么摆着）。把这些关节角喂进仿真碰撞几何，重叠的就是假碰撞。
# 实测（从真机轨迹等距抽 400 帧逐对检测重叠）：
#
#     lower_arm_link  ~ shoulder_link            105/400 帧(26%)  最大穿透 10.25mm
#     gripper_link    ~ shoulder_link             14/400 帧       最大穿透  2.95mm
#     wrist_camera_mount_link ~ wrist_link        12/400 帧       最大穿透  3.42mm
#     shoulder_link   ~ wrist_link                 5/400 帧       最大穿透  1.12mm
#
# **四分之一的真机位姿在仿真里自碰** —— 此前"重放真机轨迹总不对劲"的物理来源。
#
# 只关这 4 对，不用 ManiSkill 的 `disable_self_collisions`（那是一刀切关掉整条 articulation
# 内部所有自碰，手臂会能穿过 KIT 支架与相机柱，画面上要露馅）。
# 机制：SAPIEN 的 `collision_groups[2]` 是"忽略掩码"，两个 shape 有公共 bit 就互不碰撞。
# 每对分配一个独立 bit，互不串扰（已核对：任意非指定对的掩码交集为 0）。
FALSE_SELF_COLLISION_PAIRS = [
    ("lower_arm_link", "shoulder_link"),
    ("gripper_link", "shoulder_link"),
    ("wrist_camera_mount_link", "wrist_link"),
    ("shoulder_link", "wrist_link"),
]
# 从 20 起，避开 ManiSkill 自己用的 bit 29（`disable_self_collisions`）
_IGNORE_BIT_BASE = 20


@register_agent()
class SO101(BaseAgent):
    uid = "so101"

    # 唯一的那份几何：真机套件版，随包自包含，见 `robots/kit_assets/`。
    urdf_path = _URDF_PATH
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)  
        ),
        link=dict(
            gripper_link=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            moving_jaw_so101_v1_link=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            finger1_tip=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            finger2_tip=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    # 每个位姿都必须落在本几何的关节限位内 —— wrist_roll 只到 [−67.21°, 252.79°]。
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(kit_rest_qpos()),
            pose=sapien.Pose(q=list(euler2quat(0, 0, np.pi / 2))),
        ),
        zero=Keyframe(
            qpos=np.array([0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose(q=list(euler2quat(0, 0, np.pi / 2))),
        ),
        extended=Keyframe(
            qpos=np.array(
                [0, -0.7854, 0.7854, 0, 0, 100 * np.pi / 180]
            ),  # Fully open gripper
            pose=sapien.Pose(q=list(euler2quat(0, 0, np.pi / 2))),
        ),
    )

    arm_joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]
    gripper_joint_names = [
        "gripper",
    ]

    @property
    def _controller_configs(self):
        pd_joint_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            lower=None,
            upper=None,
            stiffness=1e3,
            damping=1e2,
            force_limit=JOINT_FORCE_LIMITS,
            normalize_action=False,
        )

        # Fast movement for SO101
        pd_joint_delta_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            [-0.1, -0.1, -0.1, -0.1, -0.1, -0.2],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
            stiffness=[1e3] * 6,
            damping=[1e2] * 6,
            force_limit=JOINT_FORCE_LIMITS,
            use_delta=True,
            use_target=False,
        )

        pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        # PD joint velocity - Not supported on real SO101
        pd_joint_vel = PDJointVelControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            lower=[-1.0, -1.0, -1.0, -1.0, -1.0, -5.0],
            upper=[1.0, 1.0, 1.0, 1.0, 1.0, 5.0],
            damping=[1e2] * 6,  
            force_limit=JOINT_FORCE_LIMITS,
            friction=0,
            normalize_action=True
        )

        controller_configs = dict(
            pd_joint_delta_pos=pd_joint_delta_pos,
            pd_joint_pos=pd_joint_pos,
            pd_joint_target_delta_pos=pd_joint_target_delta_pos,
            pd_joint_vel=pd_joint_vel,
        )
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        """关掉几何自带的 4 对假自碰撞，并给夹爪关节补上减速箱摩擦。"""
        super()._after_init()
        links = self.robot.links_map
        for k, (a, b) in enumerate(FALSE_SELF_COLLISION_PAIRS):
            bit = _IGNORE_BIT_BASE + k
            for name in (a, b):
                if name in links:
                    links[name].set_collision_group_bit(group=2, bit_idx=bit, bit=1)
        self.robot.active_joints_map["gripper"].set_friction(GRIPPER_JOINT_FRICTION)

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.finger1_link = self.robot.links_map["gripper_link"]
        self.finger2_link = self.robot.links_map["moving_jaw_so101_v1_link"]
        self.finger1_tip = self.robot.links_map["finger1_tip"]
        self.finger2_tip = self.robot.links_map["finger2_tip"]

    @property
    def tcp_pos(self):
        # computes the tool center point as the mid point between the the fixed and moving jaw's tips
        return (self.finger1_tip.pose.p + self.finger2_tip.pose.p) / 2

    @property
    def tcp_pose(self):
        return Pose.create_from_pq(self.tcp_pos, self.finger1_link.pose.q)

    def is_touching(self, object: Actor):
        """Check if the robot is touching an object """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)
        return torch.logical_or(lforce >= 1e-2, rforce >= 1e-2)

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=110):
        """Check if the robot is grasping an object (more lenient parameters)"""
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold=0.15):
        """Check if the robot is static (improved for SO101)"""
        qvel = self.robot.get_qvel()[:, :-1]  # exclude the gripper joint
        return torch.max(torch.abs(qvel), 1)[0] <= threshold
