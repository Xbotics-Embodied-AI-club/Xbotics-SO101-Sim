"""SO-101 机器人 agent，加载真机套件几何 `kit_assets/kit_v1_so101.urdf`。

本包只有这一份几何，`wrist_roll` 的零位按实物安装标定（该关节真机不做标定）。
几何里除 6 个活动关节外还有套件底板、型材框，以及两个相机的安装座与光学系
（`top_camera_optical_frame` / `wrist_camera_optical_frame`，都是 fixed 关节、不加自由度）。
相机 sensor 不在这里定义，而在 `so101_sim.envs` 里用 CameraConfig 绑到那两个光学系上。

夹爪行程与关节限位一律从这份 URDF 现读 —— 换算标度取错会静默改变夹爪百分比。
"""

import copy

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

# 真机开机位姿（度制），与两份真机数据集的首帧中位逐关节吻合 ——
# 已发布的演示每一集第 0 帧都精确等于它，逐位标准差 0.000。
# 值按本模块那套几何的关节约定给，与限位同源。换成别的起始位形，策略从没见过的
# 位姿起步，实测同一份权重成功率 0/20 → 9/10。取值与消融见 bd xb-1sc2。
REAL_HOME_DEG = [-5.76, -102.68, 92.97, 63.38, -0.53, 1.90]


def kit_rest_qpos():
    """起始位姿的弧度形式。

    Returns:
        六个关节角（弧度），即 `REAL_HOME_DEG` 的弧度形式。
    """
    return np.radians(REAL_HOME_DEG).tolist()


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
    urdf_path = str(
        Path(__file__).parent.parent
        / "kit_assets"
        / "kit_v1_so101.urdf"
    )
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
            force_limit=100,
            normalize_action=False,
        )

        # Fast movement for SO101
        pd_joint_delta_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            [-0.1, -0.1, -0.1, -0.1, -0.1, -0.2],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
            stiffness=[1e3] * 6,
            damping=[1e2] * 6,
            force_limit=100,
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
            force_limit=100,
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
        """关掉几何自带的 4 对假自碰撞，见 `FALSE_SELF_COLLISION_PAIRS` 上方的账。"""
        super()._after_init()
        links = self.robot.links_map
        for k, (a, b) in enumerate(FALSE_SELF_COLLISION_PAIRS):
            bit = _IGNORE_BIT_BASE + k
            for name in (a, b):
                if name in links:
                    links[name].set_collision_group_bit(group=2, bit_idx=bit, bit=1)

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
