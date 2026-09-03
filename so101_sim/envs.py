"""KIT 双相机任务 —— 用我们真机的 SO101（KIT 版）+ top/wrist 两路相机重建 task1 视角。

与 vendored squint 的单 base_camera 环境不同，这里的观测有**两路相机**，对齐真机数据集
`task1`（observation.images.top / observation.images.wrist，均 480×640）：

- **top**：绑在 URDF 的 `top_camera_optical_frame`（挂在 base_link 上，随支架固定俯视全局）。
- **wrist**：绑在 `wrist_camera_optical_frame`（挂在 gripper_link 上，随夹爪动、看被抓物）。

两个光学系的位姿已由真机标定写进 KIT URDF，无需 look_at 手调。挂在相机相对位姿上的是
`_OPTICAL_CONV` —— 它是两个旋转的合成：**ROS 光学系（z 前 / x 右 / y 下）→ SAPIEN 相机系
（x 前 / y 左 / z 上）** 的标准转换 `_ROS_OPTICAL_TO_SAPIEN`，再乘一个绕光轴的 180°
（`_ROLL_180`，修 URDF 里光学系不按 ROS 约定摆这件事）。

任务语义沿用 tasks/place.py 的 Place（放入料盒才算成功）：本文件只换机器人 + 相机，
不改任务逻辑。本包只有一份机器人几何（`kit_v1_so101.urdf`），`so101` 与 `so101_kit_slow`
共用它、区别只在控制器包线，所以基座朝向与 rest qpos 两个 uid 通用。
"""

import re
from pathlib import Path
from typing import ClassVar

import numpy as np
import sapien
import torch
import trimesh
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from sapien.render import RenderBodyComponent, RenderMaterial, RenderShapeTriangleMesh
from transforms3d.euler import euler2mat
from transforms3d.quaternions import qmult, quat2mat

from so101_sim.robots.so101_kit_slow import ITEM_FRICTION_RANGE, SIM_FPS
from so101_sim.tasks.place import Place

# 真机 KIT 演示套件的物体 mesh（由讲义仓 `handbook/code/assets/objects/` 的 STEP
# 用 `robots/convert_step.py` 转出，产物随包分发、STEP 原件不随包）。
_OBJECTS_DIR = Path(__file__).resolve().parent / "robots" / "kit_assets" / "objects"

# ROS 光学系 → SAPIEN 相机系的固定旋转（wxyz）：
# 相机前(x)=光学前(z)、相机左(y)=光学左(-x)、相机上(z)=光学上(-y)。
_ROS_OPTICAL_TO_SAPIEN = [0.5, 0.5, -0.5, 0.5]
# 真机相机是**正装**的（与标定一致）；仿真这条光学系约定给反了，直接渲出来的 top/wrist
# 会相对真机上下颠倒。此处绕相机光轴（SAPIEN 相机前向 x 轴）补一个 180° roll，让**仿真
# 朝实物靠拢**——即渲出与正装真机 + 标定一致的画面，而不是在图像上做后处理翻转。
_ROLL_180 = [0.0, 1.0, 0.0, 0.0]  # 绕 x 轴 180°（wxyz）
_OPTICAL_CONV = list(qmult(_ROS_OPTICAL_TO_SAPIEN, _ROLL_180))

# 相机分辨率对齐真机 task1（宽 640 × 高 480）。
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
# 相机名。**不建环境就要知道**：`lerobot-record` 在 `robot.connect()` **之前**
# 就用 `robot.observation_features` 算数据集的 features（`lerobot_record.py:502-512`），
# 那时环境还没建。名字与分辨率若只能从运行时环境读，录出来的数据集就**没有图像特征**
# —— 实测踩过：8 集只有 action / observation.state 与索引列，两路相机整个丢掉，
# 而全程不报错。真机那侧（`so_follower._cameras_ft`）也是从**配置**取形状、不从硬件取。
CAMERA_NAMES = ("top", "wrist")
# 竖直 FOV（弧度）= ChArUco 棋盘格标定结果 fovy=59.17°（RMS 0.72px，233 帧，640×480）。
# 标定视频/部署/task1 是同一路相机（同型号同镜头），故 top 与 wrist 共用此值。
# CameraConfig.fov 即 SAPIEN fovy，标定值可直接用。
CAMERA_FOV = np.deg2rad(59.17)
TOP_CAMERA_FOV = CAMERA_FOV
WRIST_CAMERA_FOV = CAMERA_FOV

# 场景外观对齐真机布置：黑色机械臂 + 黑色相机支架、白色台面。
# 机械臂与支架都是 KIT URDF 里的 link，统一走 squint 的 robot_color 旋钮刷黑；
# 其余物体（方块/圆柱/料箱）的颜色已编码进各自资产文件名，此处不动。
ROBOT_COLOR = (0.05, 0.05, 0.05)
TABLE_COLOR = (0.9, 0.9, 0.9, 1.0)

# top 相机相对 URDF 挂载光学系的位置修正（米，光学系 x 右 / y 下 / z 前）。
#
# URDF 的挂载位姿是 KIT 设计值；真机 task1 那套 rig 的实际装法与设计值有差，而标定板只能
# 标 wrist（eye-in-hand）、标不了 top（eye-to-hand），只能靠画面反解。(0,0,0) = 用设计位姿。
#
# ★**判据只落在底座安装板上。** 它与相机支架同挂 base_link ⇒ 它在图里的位置只由相机
#   位姿决定，且逐集不变，正是要标的那个量。画面里其余东西都不合格：手臂的中位剪影取决于
#   轨迹（同一串动作两侧量回的关节角最大偏差中位 9.9°）、料箱与物体逐集随机摆、下沿那条
#   深色地面带是台面尺寸差异真机没有对应物。按整幅下三成的相关峰值定值会偏 30 像素 ——
#   那一块里手臂剪影占大头。
#
# ★**只许平移，不许偏航与滚转。** 平移只改视差、不改光轴朝向，投影与真机保持平行；偏航与
#   滚转会把画面横向剪切或整体转起来。俯仰虽已放开，但平移够用就不加：留着一个必须永远
#   为 0 的旋钮本身就是陷阱。
#
# 取值靠画面反解，不靠公式算（位移到像素的换算依赖景深，实测增益约 885 px/m，而画面里
# 不同深度的东西位移量不同）：把一集的逐像素中位图与真机同一集的中位图，在安装板那块
# ROI 上做归一化互相关。扫描工具见
# `experiment_main_v1/scripts/EAI-exp-002/measure_top_offset.py`。
#
# 定稿 (-0.002, +0.027, 0)：安装板上残差 dx=0 dy=0，未平移相关 0.9787 —— 该值等于"再补一次
# 纯平移能到的上限"，即平移已到最优。x 落回设计值附近、y 差 27mm，与"支架装得比设计低"
# 一致。z 保持 0 是实测的结论而非省略：前送 30mm 只把那个上限从 0.980 抬到 0.986，后拉
# 30mm 崩到 -0.014 ⇒ 缺的从来不是缩放。
TOP_CAMERA_OFFSET = (-0.002, 0.027, 0.0)

# 复位时对 top 相机位置额外采的抖动半幅（米，同一坐标约定）。
#
# 真机自己的视角就不是一个点：modelscope 那份数据集的 9 个任务目录，底座安装板的位置
# 实测聚成 5 种装法（相对 `pick_up_a_cube`）——
#   (  +0,  +0) 5 份：battery / cube / eraser / medicine_bottle / plush_toy，逐像素相同
#                     （未平移相关 ≥0.9986）
#   ( -3, -23) 1 份：can，平移后相关 0.994 ⇒ 同一台机子挪过相机
#   (+85, +97) 1 份：Stack_the_cube_on_the_can
#   (+101,+124) 1 份：Stack_the_smaller_cube_on_the_larger_one
#   ( -74,+204) 1 份：golf
# ⇒ 仿真不该把所有集都钉在同一个视角上，而应在真机那个域内抖动，让数据自带同量级的
# 视角多样性。
#
# 取值 ±20 px（安装板深度上 885 px/m ⇒ 0.023 m）：与**混训要用的那一族**的实测跨度同量级
# （族内是纵向 23 px），也严格落在全域 `dx -74~+101 / dy -23~+204 px` 内侧。
# 不取全域跨度，是因为域不是一团连续的云而是 5 处离散装法，按包围盒均匀撒会撒到没有任何
# 真机数据的地方去。
#
# z 不抖：安装板的位移只给得出画面内的二维偏差，定不出沿光轴的量。没有实测支撑的旋钮不加。
TOP_CAMERA_JITTER = (0.023, 0.023, 0.0)


def _paint_actor(actor, rgba):
    """把一个 actor 的所有渲染面刷成纯色 rgba，并清掉原有贴图（如木纹）。"""
    for obj in actor._objs:
        entity = obj.entity if hasattr(obj, "entity") else obj
        render_body = entity.find_component_by_type(RenderBodyComponent)
        if render_body is None:
            continue
        for render_shape in render_body.render_shapes:
            for part in render_shape.parts:
                mat = part.material
                mat.set_base_color(list(rgba))
                # 清掉所有贴图，否则木纹的漫反射/法线/粗糙度会留在表面、白不彻底。
                for tex_attr in ("diffuse_texture", "normal_texture",
                                 "roughness_texture", "metallic_texture"):
                    if hasattr(mat, tex_attr):
                        setattr(mat, tex_attr, None)
                # 提高粗糙度、压低金属度，得到均匀的哑光白（不反光、无高光纹路）。
                mat.roughness = 1.0
                mat.metallic = 0.0


def _mesh_facts(glb_path):
    """读一个 glb 的包围盒全尺寸(x,y,z 米)、几何中心、以及顶点色（来自 STEP 文件名的 #RRGGBB）。"""
    m = trimesh.load(str(glb_path), force="mesh")
    rgba = np.asarray(m.visual.main_color, float) / 255.0
    return (np.asarray(m.bounding_box.extents, float),
            np.asarray(m.bounding_box.centroid, float),
            rgba)


def mesh_full_size(mesh_name):
    """物体 mesh 的真实全尺寸（米），取自 STEP 转出的 glb 包围盒。"""
    return _mesh_facts(_OBJECTS_DIR / f"{mesh_name}_visual.glb")[0]


# STEP 文件名把两个方块的颜色标反了，此处按真值覆盖渲染色（不改 STEP、不改 glb）。
# 数据要拿去训 VLA：指令说「拿蓝色方块」而画面是红的，语言接地直接错位。
# 判据与两个独立证据源见 bd xb-fmgg。圆柱的 `#9cbbd1` 与物料表一致，不动。
MESH_COLOR_OVERRIDE = {
    "cube_4": (0.612, 0.733, 0.820),      # #9cbbd1 淡蓝
    "cube_2": (0.718, 0.306, 0.302),      # #b74e4d 红
}


def _swap_visual_to_mesh(actor, glb_path, target_full_sizes, align="center",
                         mesh_rotation=None):
    """把一个 Actor 每个并行子实例的渲染网格换成 glb，碰撞盒不动。

    必须在 `_load_scene` 里调用（GPU 烘焙之前），换出的网格才进得了 GPU 渲染 ——
    与 `_paint_actor` 同一时机。碰撞盒不动意味着抓取判定与 reward 不受影响。

    颜色必须显式给 `RenderMaterial`：`RenderShapeTriangleMesh(filename=...)` 不采用
    glb 里的顶点色，缺 material 时渲出来是灰白件（实测 item 像素中位色 178,178,178
    而非 STEP 的 183,78,77）。

    Args:
        actor: 要换渲染网格的 Actor，逐个并行子实例处理。
        glb_path: STEP 转出的 glb 文件。
        target_full_sizes: 形如 (num_envs, 3) 的目标全尺寸（米），mesh 各轴缩放到与
            原碰撞盒一致。
        align: `"center"` 让 mesh 包围盒中心对到 actor 原点（方块/圆柱）；
            `"bottom"` 让 mesh 底面贴 z=0（料箱）。
        mesh_rotation: wxyz 四元数，修 mesh 建模轴与物理体轴不一致。STEP 各件建模轴
            不统一：`cube_4` 沿 z（包围盒中心 z=2cm），`cylinder_4` 沿 y（中心 y=2cm），
            而 squint 的圆柱碰撞体是绕 y 转 90° 立起来沿 z 的 ⇒ 不转就是视觉躺着、
            物理立着（实测腕部视角里圆面朝相机）。方块对称，不受影响。
            旋转在缩放与对齐之前作用于 mesh 局部系。
    """
    ext, center, rgba = _mesh_facts(glb_path)
    override = MESH_COLOR_OVERRIDE.get(Path(glb_path).stem.replace("_visual", ""))
    if override is not None:
        rgba = np.array([*override, 1.0], float)
    ext = np.where(ext < 1e-6, 1.0, ext)  # 防 0 除（薄面）
    if mesh_rotation is not None:
        # 旋转会把包围盒的轴对调，缩放与对齐都要按旋转后的量算
        rot = np.asarray(quat2mat(mesh_rotation), float)
        ext = np.abs(rot @ ext)
        ext = np.where(ext < 1e-6, 1.0, ext)
        center = rot @ center
    for i, obj in enumerate(actor._objs):
        entity = obj.entity if hasattr(obj, "entity") else obj
        old = entity.find_component_by_type(RenderBodyComponent)
        if old is not None:
            entity.remove_component(old)
        full = np.asarray(target_full_sizes[i], float)
        scale = (full / ext).astype(np.float32)
        material = RenderMaterial(base_color=rgba.tolist(), roughness=0.7, metallic=0.0)
        shape = RenderShapeTriangleMesh(filename=str(glb_path), scale=scale, material=material)
        # 缩放后 mesh 中心/底面在局部系的落点，平移回来对齐 actor 原点。
        c = center * scale
        q = list(mesh_rotation) if mesh_rotation is not None else [1.0, 0.0, 0.0, 0.0]
        if align == "bottom":
            shape.set_local_pose(sapien.Pose(p=[-c[0], -c[1], (full[2] / 2) - c[2]], q=q))
        else:
            shape.set_local_pose(sapien.Pose(p=[-c[0], -c[1], -c[2]], q=q))
        rb = RenderBodyComponent()
        rb.attach(shape)
        entity.add_component(rb)


class KitDualCameraMixin:
    """把观测相机换成绑在 KIT URDF 光学系上的 top + wrist 两路，并对齐真机的黑臂白台外观。

    可选：设 `ITEM_MESH` / `BIN_MESH` 为 kit_assets/objects 下的物体名，则把 squint 内置的
    盒/柱 item 与料箱的**渲染网格**换成真机演示套件的 STEP→mesh（碰撞盒不动，抓取/reward 不变）。
    """

    SUPPORTED_ROBOTS: ClassVar[list[str]] = ["so101", "so101_kit_slow"]

    # 子类可覆盖：把 item / bin 的视觉换成真机套件 mesh（None=沿用 squint 内置几何）。
    ITEM_MESH = None
    BIN_MESH = None
    # mesh 建模轴 → 物理体轴的修正四元数（wxyz）。None=两者一致，不用转。
    # 各 STEP 件建模轴不统一，见 `_swap_visual_to_mesh` 的说明。
    ITEM_MESH_ROTATION = None
    BIN_MESH_ROTATION = None

    def __init__(self, *args, robot_uids="so101", domain_randomization_config=None, **kwargs):
        """装好 KIT 机器人、真机起始位姿，并把机械臂与支架刷黑。

        Args:
            *args: 透传给父任务。
            robot_uids: 机器人 uid。
            domain_randomization_config: 父任务的域随机化配置，dict 或带 `.dict()` 的对象；
                这里只往里补 `robot_color`，不覆盖调用方已给的键。
            **kwargs: 透传给父任务。
        """
        # 把机械臂+相机支架刷黑：注入 robot_color 到任务的 DR 配置里，由 squint 的
        # _randomize_robot_color 统一着色（KIT URDF 里支架就是机器人的 link）。
        if domain_randomization_config is None:
            config = {}
        elif isinstance(domain_randomization_config, dict):
            config = dict(domain_randomization_config)
        else:
            config = domain_randomization_config.dict()
        config.setdefault("robot_color", list(ROBOT_COLOR))

        super().__init__(*args, robot_uids=robot_uids, domain_randomization_config=config, **kwargs)

    def _load_scene(self, options: dict):
        """搭场景，并把外观与几何对齐真机。

        必须在这一步（GPU 烘焙之前）换渲染网格与颜色，之后再换进不了 GPU 渲染。

        Args:
            options: ManiSkill 传下来的场景选项。
        """
        super()._load_scene(options)
        _paint_actor(self.table_scene.table, TABLE_COLOR)

        # `item_dimensions` / `bin_dimensions` 是每 env 的**半**尺寸，×2 才是喂给缩放的全尺寸。
        if self.ITEM_MESH is not None:
            item_full = self.item_dimensions.cpu().numpy() * 2
            _swap_visual_to_mesh(self.item, _OBJECTS_DIR / f"{self.ITEM_MESH}_visual.glb",
                                 item_full, align="center",
                                 mesh_rotation=self.ITEM_MESH_ROTATION)
        if self.BIN_MESH is not None and hasattr(self, "bin"):
            bin_full = self.bin_dimensions.cpu().numpy() * 2
            _swap_visual_to_mesh(self.bin, _OBJECTS_DIR / f"{self.BIN_MESH}_visual.glb",
                                 bin_full, align="bottom",
                                 mesh_rotation=self.BIN_MESH_ROTATION)

    @property
    def _default_sensor_configs(self):
        """两路相机的配置。宽高与 fov 是标定值，**改宽高比就等于换了一台相机**。

        Returns:
            `top` 与 `wrist` 两个 `CameraConfig`，各自挂在 KIT URDF 的光学系 link 上。
        """
        # 两路相机的朝向都只有光学系约定那一个旋转，**不额外转** —— top 与真机的偏差
        # 用 `TOP_CAMERA_OFFSET` 平移修，转相机会让投影 keystone、不再与真机平行。
        conv = sapien.Pose(q=_OPTICAL_CONV)
        top_pose = sapien.Pose(p=list(TOP_CAMERA_OFFSET), q=_OPTICAL_CONV)
        links = self.agent.robot.links_map
        return [
            CameraConfig(
                "top",
                pose=top_pose,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fov=TOP_CAMERA_FOV,
                near=0.01,
                far=100,
                mount=links["top_camera_optical_frame"],
            ),
            CameraConfig(
                "wrist",
                pose=conv,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fov=WRIST_CAMERA_FOV,
                near=0.01,
                far=100,
                mount=links["wrist_camera_optical_frame"],
            ),
        ]

    def _initialize_episode(self, env_idx, options: dict):
        """复位时把 top 相机的位置在真机视角域内重采一次。

        Args:
            env_idx: 本次复位的 env 下标。
            options: ManiSkill 传下来的复位选项。

        随机数取自 ManiSkill 逐集按 seed 播种的 `_batched_episode_rng` ⇒ 同一个 seed 重跑，
        视角一模一样，回放与跨后端复跑都可复现。

        ★ 只在**所有** env 一起复位时重采，且抖动只取 0 号 env 那一路随机流。
          ManiSkill 的 `RenderCamera.set_local_pose` 把同一个位姿写给所有子场景，
          给不了逐 env 的视角；若只有部分 env 复位就重采，会把正在跑的那些 env 的视角
          在集中途换掉。

        只动位置，不动朝向：`_OPTICAL_CONV` 原样带回去。转相机会让投影 keystone、
        不再与真机平行。
        """
        super()._initialize_episode(env_idx, options)
        if len(env_idx) != self.num_envs:
            return
        rng = self._batched_episode_rng[0]
        jitter = np.asarray(TOP_CAMERA_JITTER) * rng.uniform(-1.0, 1.0, size=3)
        position = np.asarray(TOP_CAMERA_OFFSET) + jitter
        self._sensors["top"].camera.set_local_pose(
            sapien.Pose(p=position.tolist(), q=_OPTICAL_CONV))


# 三个分发场景的物体：碰撞盒与视觉网格**同尺寸**，都钉到 STEP 实测真值，mesh 缩放恒为 1.0。
# 尺寸的唯一真相源是 STEP→glb 的包围盒（`mesh_full_size`），不在代码里另抄一份数字。
#
#   cube_4      4.0 × 4.0 × 4.0 cm      cube_2  2.0 × 2.0 × 2.0 cm
#   cylinder_4  直径 4.0 cm / 高 4.0 cm


# 真机套件的件是海绵（不是 3D 打印实心件），取 200 kg/m³ ⇒ 4cm 件约 12.8 g。
# 200 也是 squint 的默认值，它的奖励尺度与成功阈值都围绕这个量级调过 ⇒ 真值落在配方适用域内。
# 这个数改大会连锁污染整套夹持奖励与成功门，账见 bd xb-jc5o。
ITEM_DENSITY = 200.0


def _cube_size_config(mesh_name):
    """把 STEP 方块的真实边长与密度钉成 squint 的区间（上下界相同=不随机）。"""
    half = float(mesh_full_size(mesh_name)[0]) / 2
    return {"cube_half_size_range": (half, half),
            "item_density_range": (ITEM_DENSITY, ITEM_DENSITY)}


def _cylinder_size_config(mesh_name):
    """把 STEP 圆柱的真实直径/高与密度钉成 squint 的 can 区间。

    Args:
        mesh_name: `mesh_full_size` 认的件名。

    Returns:
        squint 的 can 尺寸配置字典。

    Raises:
        ValueError: 横截面不是圆 —— 那说明这个件的建模轴不是 y，按 y 取高会取错。
    """
    ext = mesh_full_size(mesh_name)
    # 建模轴沿 y（见 `_swap_visual_to_mesh` 的 `mesh_rotation` 说明），所以 x/z 是直径、y 是高。
    # cube_4 沿 z 而 cylinder_4 沿 y，STEP 各件不统一，取错轴时尺寸会静默变成另一个数。
    if abs(float(ext[0]) - float(ext[2])) > 1e-3:
        raise ValueError(f"{mesh_name} 的横截面不是圆：x={ext[0]:.5f} z={ext[2]:.5f}")
    half_radius = float(ext[0]) / 2
    half_height = float(ext[1]) / 2
    return {"can_radius_range": (half_radius, half_radius),
            "can_half_height_range": (half_height, half_height),
            "item_density_range": (ITEM_DENSITY, ITEM_DENSITY)}


# 撒点区照抄真机实测分布，不照抄 IK 可达性扫描（那条曾给出作废的假阴性，见 bd xb-oy5i）。
# 依据 ModelScope cube 任务 174 集反解出的末端位置：物体 x p5..p95 = 0.172..0.368、
# y = -0.099..0.190；真机 y 明显偏正（中位 +0.042~+0.053），故中心不取 y=0。
SPAWN_BOX_POS = [0.27, 0.045]
SPAWN_BOX_HALF_SIZE = 0.10


class ReachableSpawnMixin:
    """把物体/料箱的撒点区对齐真机实测分布。"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("spawn_box_pos", list(SPAWN_BOX_POS))
        kwargs.setdefault("spawn_box_half_size", SPAWN_BOX_HALF_SIZE)
        super().__init__(*args, **kwargs)


class RealSizeItemMixin:
    """把 item 的碰撞尺寸钉到 `ITEM_MESH` 对应 STEP 件的真实尺寸。

    squint 的 Lift 用 `domain_randomization_config` 的尺寸区间造碰撞体，这里把区间上下界
    都设成 STEP 真值即可——不改 vendored 任务逻辑，reward/success 判定原样沿用。
    """

    ITEM_SIZE_CONFIG: ClassVar[dict[str, tuple[float, float]]] = {}

    def __init__(self, *args, domain_randomization_config=None, **kwargs):
        """把 `ITEM_SIZE_CONFIG` 里的尺寸区间补进域随机化配置。

        Args:
            *args: 透传给父任务。
            domain_randomization_config: 父任务的配置；只补本类声明的键，
                调用方已给的不覆盖。
            **kwargs: 透传给父任务。
        """
        if domain_randomization_config is None:
            config = {}
        elif isinstance(domain_randomization_config, dict):
            config = dict(domain_randomization_config)
        else:
            config = domain_randomization_config.dict()
        for key, value in self.ITEM_SIZE_CONFIG.items():
            config.setdefault(key, value)
        super().__init__(*args, domain_randomization_config=config, **kwargs)


# `cylinder_4` 的 STEP 建模轴沿 y（包围盒中心 y=2cm），而 squint 的圆柱碰撞体立着沿 z
# ⇒ 视觉必须绕 x 转 -90° 把 y 轴掰到 z 轴，否则渲出来是躺着的圆柱（实测过）。
_CYLINDER_Y_TO_Z = [np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0.0, 0.0]


# 「真夹住」用**几何贴合**判，不用接触力：PhysX 的 `contact_offset = 0.02` 意味着两体
# 相距 2 cm 以内就报接触力 ⇒ 力判据天然分不清「夹住」与「靠近」。实测有 19.8% 的帧
# 活动爪根本没碰到方块，靠固定爪单侧顶着 + 摩擦带走。判据、三条阈值各自的账、
# 以及「reward 朝理想值优化、门取可达值」这条结论，见 bd xb-4zle。
FIRM_GRASP_GAP_TOL = -0.002       # reward 满分阈值（米）：需压入 2 mm。设 0 会让策略卡在边界抖
FIRM_GRASP_GAP_SCALE = 0.006      # 从满分阈值再放宽 6 mm 处夹持分归零
PINCH_EDGE_MARGIN = 0.70          # 接触点面内归一化位置上限（1.0=贴棱）；超过即不算夹住
# reward 用的连续量：两爪接触点中点到质心的归一化距离
PINCH_CENTER_GOOD = 0.35          # 到此为止算「夹住质心」，给满分
PINCH_CENTER_SCALE = 0.55         # 再远这么多后中部分归零
# success 门比 reward 的满分线**宽** —— 门设在满分线上会够不着，正样本采不到。
GATE_CENTER_OFF = 0.50
GATE_GAP = 0.0
# 松手瞬间物体的速度上限（m/s）。真机是放到底、停稳再张爪，物体此时基本不动；
# 取 5 cm/s 作为「已经躺住」的判据，横扫中途松手远超此值。
GENTLE_RELEASE_SPEED = 0.05
# 方块静置在箱底时的绝对高度（米）。实测落到箱底后 item z 稳定在 2.50 cm（桌面上是 2.00 cm）。
# 料箱只在 xy 随机、z 不变，所以这是个常量。
ITEM_REST_Z_IN_BIN = 0.025
# 「往箱底放」的引导。判据加了而引导没加，策略永远撞不到那道墙 —— 所以这几个必须在。
LOWER_REWARD_SCALE = 2.0
LOWER_REWARD_K = 20.0        # 高出箱底 5 cm 得 0.24 分、1 cm 得 0.80 分、贴底 1.00 分
RELEASE_HIGH_PENALTY = 2.0   # 高处或带速度松手的惩罚
JAW_LINKS = ("gripper_link", "moving_jaw_so101_v1_link")
JAW_SAMPLE_STRIDE = 24            # 爪网格顶点抽稀步长（每爪约几百点，够用且够快）


_JAW_VERTEX_CACHE = {}


def _jaw_sample_points(urdf_path, link_name):
    """某爪 link 的碰撞网格顶点（link 局部系，抽稀）。按 URDF 的 collision origin 变换。"""
    key = (str(urdf_path), link_name)
    if key in _JAW_VERTEX_CACHE:
        return _JAW_VERTEX_CACHE[key]

    text = Path(urdf_path).read_text()
    block = re.search(rf'<link name="{link_name}">(.*?)</link>', text, re.DOTALL)
    chunks = []
    for col in re.findall(r"<collision>(.*?)</collision>", block.group(1), re.DOTALL):
        origin = re.search(r'<origin xyz="([^"]+)"(?:\s+rpy="([^"]+)")?', col)
        mesh_ref = re.search(r'<mesh filename="([^"]+)"', col)
        if not mesh_ref:
            continue
        mesh_path = Path(urdf_path).parent / mesh_ref.group(1)
        if not mesh_path.exists():
            continue
        xyz = (np.array([float(x) for x in origin.group(1).split()])
               if origin else np.zeros(3))
        rpy = (np.array([float(x) for x in origin.group(2).split()])
               if (origin and origin.group(2)) else np.zeros(3))
        verts = np.asarray(trimesh.load(mesh_path, force="mesh").vertices,
                           float)[::JAW_SAMPLE_STRIDE]
        chunks.append((euler2mat(*rpy) @ verts.T).T + xyz)

    pts = np.concatenate(chunks)
    _JAW_VERTEX_CACHE[key] = pts
    return pts


class FirmGraspRewardMixin:
    """给 Place 的 reward 加一项「几何贴合」，并让搬运分以真夹住为前提。

    **为什么必须改 reward**：squint 的 Place 成功判据是
    `物体在料箱 x/y 内 ∧ 已松手 ∧ 物体静止 ∧ 机器人静止` —— **不含 `is_item_grasped`**。
    于是"托着走到箱口再丢进去"完全满足判据，策略没有任何理由把爪子收紧。

    **为什么判据必须是几何而不是力**：PhysX `contact_offset = 0.02` ⇒ 两体相距 2 cm 以内
    就报接触力，所以力大小完全不能代表「夹住」。实测两指各 16–18 N 的同时，活动爪
    有 19.8% 的帧根本没碰到方块。判据的完整账见 bd xb-4zle。

    三处改动都在子类，不碰 vendored：

    1. 几何贴合奖励：两爪各取碰撞网格采样点，算到方块（有向盒）的有符号距离，
       取两爪中较差那一侧；≤0（压入）给满分，缝 `FIRM_GRASP_GAP_SCALE` 以上归零。
    2. 搬运分以夹住为前提：vendored 的 `is_item_above_bin` 分支会**无条件**覆盖 reward、
       不要求 grasped，这正是「托着走」能拿高分的口子。未曾真贴合过则扣掉该增量。
    3. `evaluate()` 导出 `jaw_gap` / `is_firm_grasp` 供验收与常驻断言使用。
    """

    def _jaw_points(self):
        """两爪采样点（link 局部系），首次调用时按 URDF 解析并缓存。"""
        if getattr(self, "_jaw_pts", None) is None:
            urdf = self.agent.urdf_path
            self._jaw_pts = {
                name: torch.as_tensor(_jaw_sample_points(urdf, name),
                                      dtype=torch.float32, device=self.device)
                for name in JAW_LINKS
            }
        return self._jaw_pts

    def _pinch_facts(self):
        """两爪的贴合事实：不只「压得多深」，还有「压在哪」。

        只看 `gap` 会瞎：它取所有采样点有符号距离的最小值，`argmin` 把「接触发生在哪儿」
        整个丢掉 —— 棱上蹭到 1 mm 与侧面中部压入 1 mm 同分。实测某一版搬运帧里
        `gap≤0` 占 84.8%，但贴棱（面内偏移 >0.7）的帧占 96.1%，真·两侧夹持只有 0.7%：
        方块是靠擦棱加摩擦被带走的，目视像吸附。

        Returns:
            `(较差 gap, 两爪是否落在一对相对面, 较差的面内偏移, 接触中点到质心的归一化距离)`。
            面内偏移取除法向轴外两个归一化坐标的较大者，1.0 = 贴棱。
        """
        Ti = self.item.pose.to_transformation_matrix()
        Ri, pi = Ti[:, :3, :3], Ti[:, :3, 3]
        half = self.item_dimensions                      # (N,3) 半尺寸
        gaps, axes, signs, offs, norms = [], [], [], [], []
        for name, pts in self._jaw_points().items():
            T = self.agent.robot.links_map[name].pose.to_transformation_matrix()
            world = torch.einsum("nij,vj->nvi", T[:, :3, :3], pts) + T[:, :3, 3][:, None]
            local = torch.einsum("nij,nvi->nvj", Ri, world - pi[:, None])
            over = (local.abs() - half[:, None, :])
            outside = over.clamp(min=0).norm(dim=-1)     # 盒外欧氏距离
            inside = over.max(dim=-1).values             # 盒内为负
            signed = torch.where(outside > 0, outside, inside)
            best, idx = signed.min(dim=-1)
            pick = local[torch.arange(local.shape[0], device=local.device), idx]
            norm = pick / half                           # (N,3) 归一化到 ±1
            ax = norm.abs().argmax(dim=-1)
            sg = torch.sign(norm.gather(1, ax[:, None]).squeeze(1))
            on_axis = torch.nn.functional.one_hot(ax, norm.shape[-1]).bool()
            off = norm.abs().masked_fill(on_axis, -1.0).max(dim=-1).values
            gaps.append(best), axes.append(ax), signs.append(sg)
            offs.append(off), norms.append(norm)

        gap = torch.maximum(gaps[0], gaps[1])
        opposite = (axes[0] == axes[1]) & (signs[0] == -signs[1])
        # ★给 reward 用的连续量：两爪接触点的**中点**到方块质心的归一化距离。
        # 它一个数同时表达「相对」与「中部」——两爪都蹭在顶棱时中点偏到顶部（大），
        # 真从两侧夹住时中点≈质心（小）。比把 `opposite` 当硬门好：硬门早期恒假，
        # 夹持信号会稀疏到学不动。
        center_off = ((norms[0] + norms[1]) / 2).norm(dim=-1)
        return gap, opposite, torch.maximum(offs[0], offs[1]), center_off

    def _jaw_gap(self):
        """两爪到方块盒的最小有符号距离，取较差那一侧。<=0 = 两爪都贴合（**不含方位**）。"""
        return self._pinch_facts()[0]

    def compute_dense_reward(self, obs, action, info):
        """父类 reward 加三项：几何贴合、往箱底放、以及对高处松手的惩罚。

        Args:
            obs: 当前观测。
            action: 刚执行的动作。
            info: `evaluate()` 的产物；含 `jaw_gap` 等几何量时直接复用，避免重算。

        Returns:
            每个并行环境一个标量 reward。
        """
        reward = super().compute_dense_reward(obs, action, info)

        # 只取这两项：`_pinch_facts` 还给「是否落在一对相对面」与「面内偏移」，
        # 但 reward 用的是 gap 与到质心的归一化距离，另两项由 `evaluate()` 自己用。
        if "pinch_center_off" in info:
            gap, center_off = info["jaw_gap"], info["pinch_center_off"]
        else:
            gap, _, _, center_off = self._pinch_facts()
        # 贴合分：gap<=tol 满分，线性衰减到 tol+scale 归零
        grip_score = (1.0 - (gap - FIRM_GRASP_GAP_TOL) / FIRM_GRASP_GAP_SCALE).clamp(0.0, 1.0)
        # 必须再乘「方位」项，否则策略会去优化棱边接触：接触点离棱越远越高分。
        centrality = (1.0 - (center_off - PINCH_CENTER_GOOD)
                      / PINCH_CENTER_SCALE).clamp(0.0, 1.0)
        grip_score = grip_score * centrality

        # 这里必须用**可达**的门判据。下面 `above_bin_no_firm` 会按 `~_ever_firm` 每步扣
        # 1.5 分，而「举到料箱上方」是完成任务的必经环节 —— 判据若几乎不可能为真
        # （严格版实测仅 0.2% 的帧满足），这一项就退化成对放置动作本身的恒定惩罚，
        # 策略学到的是别把物体举过去。收严这个判据而不回头看它的下游用途，
        # 曾把成功率压到 0.062。见 bd xb-4zle。
        firm_now = (gap <= GATE_GAP) & (center_off <= GATE_CENTER_OFF)
        if getattr(self, "_ever_firm", None) is None or self._ever_firm.shape[0] != firm_now.shape[0]:
            self._ever_firm = torch.zeros_like(firm_now)
        self._ever_firm |= firm_now

        # 贴合分只在**搬运途中**给。到了料箱上方就该松手，此时若还按住不放继续发分，
        # 举着每步稳拿 4+place+static+1≈7~8，而松手要先穿过物体下落、尚未静止的低分段
        # 才够得着 success 的 9 —— 在 gamma=0.96（视野约 25 步）下折算回来还不如举着。
        # 策略于是学出「夹稳后悬在箱口不放」：实测 331 帧不松手、jaw_gap 一路收紧到 -1.2mm。
        # 把箱上方那段的贴合分去掉，松手才是严格更优。
        grasped = info["is_item_grasped"]
        carrying = grasped & (~info["is_item_above_bin"])
        reward = reward + grip_score * carrying.float()
        above_bin_no_firm = info["is_item_above_bin"] & (~self._ever_firm)
        reward = reward - 1.5 * above_bin_no_firm.float()

        # 放置引导：夹着物体到了箱上方后，越把物体往箱底放分越高。真机就是这么做的 ——
        # 放到底、停稳、才张爪，全程不脱手（300 集实测松手前速度只有全程中位的 0.11）。
        # 只把「轻放」写进 success 而不给梯度是不够的：那一版 12/12 条成功轨迹全部在离箱底
        # 3.8–9.2 cm 的空中以 0.135–0.540 m/s 甩手 ⇒ 不是采样不足，是没给引导。
        above = info["is_item_above_bin"]
        holding = info["is_item_grasped"]
        height = (self.item.pose.p[:, 2] - ITEM_REST_Z_IN_BIN).clamp(min=0.0)
        lower_score = 1.0 - torch.tanh(LOWER_REWARD_K * height)
        reward = reward + LOWER_REWARD_SCALE * lower_score * (above & holding).float()

        # 高处或带速度松手要扣分，否则"举到箱上方直接丢"仍是捷径
        item_speed = torch.linalg.norm(self.item.linear_velocity, dim=-1)
        bad_release = above & (~info["robot_touching_item"]) & (
            (height > 0.01) | (item_speed > GENTLE_RELEASE_SPEED))
        reward = reward - RELEASE_HIGH_PENALTY * bad_release.float()
        return reward

    def _initialize_episode(self, env_idx, options):
        """开新一集，并清掉「本集曾真夹住」这两个跨集状态。

        Args:
            env_idx: 本次被 reset 的并行环境下标 —— **只清这些**，别的还在进行中。
            options: ManiSkill 传下来的初始化选项。
        """
        super()._initialize_episode(env_idx, options)
        if getattr(self, "_ever_firm", None) is not None:
            self._ever_firm[env_idx] = False
        # 「本集曾真夹住」是 success 的必要条件，跨集必须清零，否则一集成功之后
        # 后面每集都白送这个条件（并行环境下尤其隐蔽：只有被 reset 的那几个 env 该清）
        if getattr(self, "_ever_center_grasp", None) is not None:
            self._ever_center_grasp[env_idx] = False

    def evaluate(self):
        """父类的判定，外加几何贴合量与「本集曾真夹住」这个必要条件。

        Returns:
            父类 info 加上 `jaw_gap` / `pinch_center_off` / `is_pinch_opposite` /
            `pinch_edge_off` 与收紧后的 `success`。几何量导出来供验收与常驻断言用，
            也让 `compute_dense_reward` 不必重算一遍。
        """
        info = super().evaluate()
        gap, opposite, edge_off, center_off = self._pinch_facts()
        info["jaw_gap"] = gap
        info["pinch_center_off"] = center_off
        info["is_pinch_opposite"] = opposite
        info["pinch_edge_off"] = edge_off
        # 旧口径：只问压没压到，不问压在哪（保留供对照，**不要拿它当"夹住了"**）
        info["is_firm_grasp"] = gap <= FIRM_GRASP_GAP_TOL
        # 报告口径：压入 ∧ 落在一对相对面 ∧ 都在面中部。
        # ⚠️ 只作报告，**不进 success**：面归属是 `argmax|坐标|` 定的，接触点靠近棱时
        # 3mm 的扰动就会翻面（可达性扫描实测 True/False 来回跳）⇒ 拿它当门会把噪声当信号。
        info["is_true_pinch"] = (info["is_firm_grasp"] & opposite
                                 & (edge_off <= PINCH_EDGE_MARGIN))
        # ★进 success 的口径用**连续量**：压入够深 ∧ 两爪接触点中点靠近方块质心。
        # 不含离散的面归属，故不抖；"中点靠近质心"本身已排除「两爪都蹭同一条棱」。
        # ★「放到底再松手」：真机 300 集实测，松手前 10 帧速度只有全程中位的 **0.11**、
        # 89% 的集明显减速；腕部画面里松手前 20 帧方块**已经躺在箱底**，前后几乎不动。
        # 即真机是「移到箱上方 → 一直放到底 → 停稳 → 张爪」，全程不脱手，**没有下落这一段**。
        # 我们的策略却是夹着横扫、边走边松手（实测水平位移 9.5cm > 落差 6.4cm，
        # "下落"段仍有 59% 的帧夹着，等效 g 仅 0.12–1.29 m/s²）。
        # 判据取**松手瞬间物体的速度**：放到底再张爪则≈0，横扫中途松手则很大。
        # 用速度而非高度，可避开料箱几何、也不必猜静置高度。
        item_speed = torch.linalg.norm(self.item.linear_velocity, dim=-1)
        info["item_speed"] = item_speed
        info["is_gentle_release"] = (~info["robot_touching_item"]) & (
            item_speed <= GENTLE_RELEASE_SPEED)

        center_grasp = (gap <= GATE_GAP) & (center_off <= GATE_CENTER_OFF)
        info["is_center_grasp"] = center_grasp
        if getattr(self, "_ever_center_grasp", None) is None \
                or self._ever_center_grasp.shape[0] != center_grasp.shape[0]:
            self._ever_center_grasp = torch.zeros_like(center_grasp)
        self._ever_center_grasp |= center_grasp

        # ★把「真夹住」变成成功的必要条件。
        # 不加这条，真夹住永远是可选项：squint 的 Place success =
        # 物体在箱内 ∧ 已松手 ∧ 静止，擦着棱把方块蹭进箱子照样满分，而夹持只是每步
        # 最多 +1 的附加奖励、抵不过省力路径。实测 v9（只改 reward 不改 success）
        # 搬运期 center_off 中位仍有 0.80、真夹持率 1.3% —— 与只改 reward 前几乎没变。
        # 同 `ReturnHomeMixin` 把「收回起始位姿」写进 success 的道理。
        info["placed"] = info["success"]          # 父类口径：放进去了（不问怎么夹的）
        # 成功 = 放进去了 ∧ 曾真夹住 ∧ 松手时物体已躺住（不是甩进去的）
        info["success"] = (info["success"] & self._ever_center_grasp
                           & info["is_gentle_release"])
        return info


class ReturnHomeMixin:
    """让一集包含「放完手再收回起始位姿」，策略自己学，不靠事后合成。

    真机采的 `task1` 每集都是从起始位姿出发、干完活回到起始位姿收尾（首末关节角最大只差
    11.8°）。而 squint 的 Place 判成功只要
    `物体在箱内 ∧ 没碰物体 ∧ 机器人静止 ∧ 没碰箱` —— **只要求"停住"，不要求"停在哪"**，
    停在料箱正上方照样算成功。实测出来的轨迹首末差中位 48.9°，机械臂就悬在箱口。

    三处改动，缺一不可：

    1. **成功判据加「回到起始位姿」**。不加这条，回家永远是可选项。
    2. **撤掉放手后那份「不动就给分」**。父类在 `is_item_above_bin` 段有
       `static_robot_reward = 1 - tanh(10|qvel|)`，停着不动稳拿满分；不撤掉它，
       启动回家反而先掉分。
    3. **回家要给稠密梯度，不能只靠终点大奖励**。gamma=0.96 视野约 25 步，而回家要走
       六七十步，终点那份奖励折算回放手时刻已经所剩无几（0.96^70≈0.06）。所以按
       「离起始位姿多近」逐帧给分，每挪一步都立刻变好，策略才跟得住。

    ★起始位姿取**每集 reset 后的实际 qpos**，不取 `start` keyframe 的名义值：两者对不上
    （keyframe 的 `wrist_roll` 是 −90°，实测稳定在 −67.2°，差 22.8°），拿名义值当目标会让
    「回到起始位姿」这句话在数据上不成立。
    """

    HOME_TOL = 0.15            # rad，关节空间 L2 距离（不含夹爪）；约合每关节几度
    HOME_REWARD_SCALE = 3.0    # 回家进度分的权重，要压过撤掉的那份 static 分
    SUCCESS_REWARD = 14.0      # 父类成功给 9；回家更长更难，终点分要相应抬高

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        qpos = self.agent.robot.get_qpos()
        if getattr(self, "_home_qpos", None) is None or self._home_qpos.shape != qpos.shape:
            self._home_qpos = qpos.clone()
        self._home_qpos[env_idx] = qpos[env_idx]

    def _latch_reachable_home(self):
        """把起始位姿改记成**第一步之后**的位姿。

        ★reset 那一刻的 qpos 不是物理可达的稳态：`start` keyframe 给的
        `wrist_roll = −90°`，第一个控制步就被拉到 −67.2° 并从此不动（实测一步跳 22.9°，
        零动作静置 60 步也回不去）。拿 reset 帧当回家目标，`home_dist` 有个 0.40 rad 的
        永久下界，远超 0.15 的容差 —— 机器人站着不动都算没回家，success 恒为 0。
        实测就是这么栽的：822k 步 `success_once` 严格 0.00，不是学不会，是够不着。
        """
        first_step = self.elapsed_steps == 1
        if bool(first_step.any()):
            qpos = self.agent.robot.get_qpos()
            self._home_qpos[first_step] = qpos[first_step]

    def _home_distance(self):
        """当前关节角到本集起始位姿的距离。夹爪不算——它开合由放手决定，不代表姿态回没回。"""
        qpos = self.agent.robot.get_qpos()
        return torch.linalg.norm(qpos[:, :-1] - self._home_qpos[:, :-1], dim=-1)

    def evaluate(self):
        """父类的判定，外加「回到 home」这一段的量。

        Returns:
            父类 info 加上 `home_dist` / `is_robot_home` / `placed`。`placed` 保留父类
            口径（放进去了，不管停在哪），`success` 则要求同时回到 home。
        """
        info = super().evaluate()
        self._latch_reachable_home()
        home_dist = self._home_distance()
        info["home_dist"] = home_dist
        info["is_robot_home"] = home_dist <= self.HOME_TOL
        info["placed"] = info["success"]          # 父类口径：放进去了（不管停在哪）
        info["success"] = info["success"] & info["is_robot_home"]
        return info

    def compute_dense_reward(self, obs, action, info):
        reward = super().compute_dense_reward(obs, action, info)

        # 放开手之后才谈回家；还夹着就回去等于把物体带走
        released = info["is_item_above_bin"] & (~info["robot_touching_item"])

        robot_v = torch.linalg.norm(self.agent.robot.get_qvel()[:, :-1], axis=1)
        static_robot_reward = 1 - torch.tanh(robot_v * 10)
        home_score = 1 - torch.tanh(2.0 * info["home_dist"])

        # 撤掉「停着不动」那份，换成「离家多近」那份：停在原地不再有收益，往回挪立刻加分。
        # ★这里按 `released` 给、不按「还没到家」给：若一进容差圈就把这项撤掉，机械臂
        #   刚够到家、还没停稳（够不着 success）的那几帧会平白掉两分，成了回家路尽头的
        #   一个坑，策略宁可停在坑外。实测过一版就是这么掉的（9.33 -> 7.47）。
        adjust = self.HOME_REWARD_SCALE * home_score - static_robot_reward
        reward = reward + adjust * released.float()

        reward[info["success"]] = self.SUCCESS_REWARD
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / self.SUCCESS_REWARD


class SlowRobotMixin:
    """把任务的机器人换成真机速度包线版，并提高物体摩擦（夹得更稳）+ 对齐真机 30fps。

    squint 的任务 `__init__` 只在 `robot_uids` 是 `so100`/`so101` 时才设 `base_z_rot` 与
    `rest_qpos`，换成别的 uid 会在 `_load_agent` 里报 `no attribute 'base_z_rot'`；
    这里像 `KitDualCameraMixin` 一样显式补上。

    `control_freq` 设成 `SIM_FPS`（30）以对齐真机 task1；delta 上限在
    `so101_kit_slow.per_step_limits` 里按同一个 `SIM_FPS` 反算，故速度不随帧率漂。
    """

    SLOW_ROBOT_UID = "so101_kit_slow"

    def __init__(self, *args, robot_uids=None, domain_randomization_config=None, **kwargs):
        """装好真机速度包线版机器人，并按几何选对应的起始位姿。

        Args:
            *args: 透传给父任务。
            robot_uids: 忽略，一律用 `SLOW_ROBOT_UID` —— 速度包线是机器人自身的属性，
                让调用方换 uid 就等于绕过了包线。
            domain_randomization_config: 父任务的配置；只补摩擦区间等本类声明的键。
            **kwargs: 透传给父任务。
        """
        if domain_randomization_config is None:
            config = {}
        elif isinstance(domain_randomization_config, dict):
            config = dict(domain_randomization_config)
        else:
            config = domain_randomization_config.dict()
        # 提高物体摩擦：3D 打印件配橡胶指垫，squint 默认 (0.1,0.5) 偏低导致"托着走"
        config.setdefault("item_friction_range", list(ITEM_FRICTION_RANGE))

        super().__init__(*args, robot_uids=self.SLOW_ROBOT_UID,
                         domain_randomization_config=config, **kwargs)

    @property
    def _default_sim_config(self):
        """把控制频率提到 30Hz 对齐真机 task1。

        ★`control_freq` 不是构造参数，走 `SimConfig`；且 **`control_freq` 必须整除
        `sim_freq`**（`sapien_env` 里有 assert）。默认 `sim_freq=100` 除不尽 30，
        故同时把 `sim_freq` 提到 120（每个控制步 4 个物理子步，物理步长 1/120s）。
        """
        cfg = super()._default_sim_config
        cfg.sim_freq = 120
        cfg.control_freq = SIM_FPS
        return cfg


# 训练与重渲必须共用同一套几何 —— 换一套几何等于换一套 `wrist_roll` 零位，同一个 qpos
# 把夹爪放到不同位置（实测 `gripper_link` 差 6.0 mm、`moving_jaw` 差 38.1/13.7 mm），
# 策略按一套学会夹紧、验收按另一套算，永远对不上。别再引入第二套。账见 bd xb-1sc2。
#
# 双相机发 6 通道而 squint 专家的 CNN 编码器只吃 3 通道，
# 所以专家训练用单相机，双相机只在重渲时用。


class KitGeomSingleCamMixin(SlowRobotMixin):
    """真机速度包线机器人 + squint 默认单 base_camera（给专家训练/rollout 用）。

    相机沿用父环境默认，故观测是 3 通道，与专家编码器匹配。
    """

    SUPPORTED_ROBOTS: ClassVar[list[str]] = ["so101", "so101_kit_slow"]
    SLOW_ROBOT_UID = "so101_kit_slow"


# 三个分发场景走 squint 原生的 reward 与 success，不套 `FirmGraspRewardMixin`：
# 那套几何夹持奖励的前提是「件重 80 g、指尖托不住」，而件其实是海绵（见 `ITEM_DENSITY`）。
# 前提不成立，补丁就只剩副作用 —— 每加一条约束成功率塌一个量级，而 squint 原配方
# 在其适用域内能到 1.00。账见 bd xb-jc5o。
#
# 起止的 home 段不由策略学：那两段是两个已知关节构型之间的过渡，关节空间插值经控制器
# 走出来即可，不需要逆解。让策略学回家会把 400 步预算耗在收臂上，取放反而练不成。
#
# 20 mm 方块与 40 mm 圆柱都在夹爪行程内（削薄后两指净空 −10°→2.4 mm、20°→42.1 mm）；
# 圆柱走 squint 的 `item_type="can"` 分支。
class KitSlowMixin(SlowRobotMixin, KitDualCameraMixin):
    """KIT 双相机 + 真机速度包线 + 高摩擦 + 30fps（重渲用）。

    继承 `SlowRobotMixin` 拿到摩擦/帧率/属性补齐三件事，只把机器人 uid 换成 KIT 版。
    """

    SUPPORTED_ROBOTS = KitDualCameraMixin.SUPPORTED_ROBOTS + ["so101_kit_slow"]
    SLOW_ROBOT_UID = "so101_kit_slow"


# ══ 对外分发环境：一个场景一个环境，就这三个 ═══════════════════════════════
#
# 命名 `SO101<动词><物体><尺寸mm>-v<版本>`，名字里不带开发沿革。每个场景只有一个环境、
# 带完整双相机（top + wrist），**录制与渲染共用它** —— 于是不存在「孪生环境是否逐字同构」
# 这种要另外验的风险。
#
# 成功判据用 squint 原逻辑，四条的合取（`place.py` 的 `evaluate`）：物体落在料箱 x/y 范围内
# ∧ 手臂不碰物体 ∧ 手臂不碰料箱 ∧ 机器人静止。**不含"物体静止"** —— `is_item_static` 算了
# 但没进 `success`。也就是「东西进箱了、手撤开了、臂停稳了」，用画面自证比在腕部视角里辨接触可靠。
#
# 三个场景共用 `bin_2` 料箱（80×100×30 mm，与 STEP 包围盒逐字一致，所以不覆盖 bin 尺寸）；
# 物体碰撞尺寸钉到各自 STEP 真值，不随机。
@register_env("SO101PickPlaceCube40-v1", max_episode_steps=400)
class SO101PickPlaceCube40(KitSlowMixin, RealSizeItemMixin, ReachableSpawnMixin, Place):
    """把 40mm 方块抓起来放进料箱。top + wrist 双相机，真机速度包线，真值资产。"""

    ITEM_MESH = "cube_4"
    BIN_MESH = "bin_2"
    ITEM_SIZE_CONFIG = _cube_size_config("cube_4")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="cube", **kwargs)


@register_env("SO101PickPlaceCube20-v1", max_episode_steps=400)
class SO101PickPlaceCube20(KitSlowMixin, RealSizeItemMixin, ReachableSpawnMixin, Place):
    """把 20mm 方块抓起来放进料箱。top + wrist 双相机，真机速度包线，真值资产。"""

    ITEM_MESH = "cube_2"
    BIN_MESH = "bin_2"
    ITEM_SIZE_CONFIG = _cube_size_config("cube_2")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="cube", **kwargs)


@register_env("SO101PickPlaceCylinder40-v1", max_episode_steps=400)
class SO101PickPlaceCylinder40(KitSlowMixin, RealSizeItemMixin, ReachableSpawnMixin,
                               Place):
    """把 40mm 圆柱抓起来放进料箱。top + wrist 双相机，真机速度包线，真值资产。"""

    ITEM_MESH = "cylinder_4"
    BIN_MESH = "bin_2"
    # 圆柱 STEP 件的建模轴是 y，物理体的轴是 z。缺这一行 ⇒ 视觉网格躺着渲染、
    # 物理体却立着，画面上就是「圆柱倒在桌上」。
    ITEM_MESH_ROTATION = _CYLINDER_Y_TO_Z
    ITEM_SIZE_CONFIG = _cylinder_size_config("cylinder_4")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="can", **kwargs)
