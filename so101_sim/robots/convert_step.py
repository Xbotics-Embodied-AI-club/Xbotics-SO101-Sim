"""资产工具：把真机任务物体的 STEP（CAD）转成仿真可用的 mesh。

真机套件的物体给的是 STEP 格式：两个方块、一个圆柱、一个黑 bin。
仿真要用它们得先转成网格 —— cascadio（trimesh 的 OpenCASCADE 后端）读 STEP，导出：

- `<name>_visual.glb`：可视网格（带文件名 hex 颜色）。
- `<name>_collision.obj`：凸包碰撞网格（sapien/PhysX 只吃凸体；bin 这类内凹件先用凸包，
  真正做"放进 bin"的凹腔碰撞是后续任务的事）。

文件名编码信息：`#RRGGBB` 是颜色（在文件名任意位置，取不到一律回落黑色）；方块与圆柱的
`_N`（如 `cube_4`）是边长/直径档，单位厘米，而 `bin_2` 的 `_2` 是版本号不是尺寸
（它实测 80×100×30 mm）。产物落同目录 `kit_assets/objects/`，随包自包含。

**这是一次性资产工具，不在安装后的运行路径上。** 输入的 STEP 原件**不随本包分发** ——
它们在讲义仓 `handbook/code/assets/objects/`，本包是独立 submodule，只在源码检出布局下
够得着。产物已经在 `kit_assets/objects/` 里，所以正常使用本包时不需要跑它；
`SRC` 不存在时 `convert_all()` 会把路径打出来并抛 `FileNotFoundError`。
"""

import re
from pathlib import Path

import numpy as np
import trimesh

SRC = Path(__file__).resolve().parents[4] / "handbook" / "code" / "assets" / "objects"
DST = Path(__file__).parent / "kit_assets" / "objects"


def _load_merged(step_path: Path) -> trimesh.Trimesh:
    """STEP 里可能是多实体 scene，合并成单个 Trimesh。"""
    loaded = trimesh.load(step_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
        return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    return loaded


def _color_from_name(name: str):
    """从文件名末尾的 #RRGGBB 取颜色；黑 bin 文件名是中文『黑色』，回落黑色。"""
    m = re.search(r"#([0-9a-fA-F]{6})", name)
    if m:
        h = m.group(1)
        return [int(h[i : i + 2], 16) for i in (0, 2, 4)] + [255]
    return [20, 20, 20, 255]  # 黑 bin


def convert_all() -> None:
    """把 `SRC` 下所有 STEP 转成可视 glb 与凸包碰撞 obj，落到 `DST`。

    每个件打印包围盒全尺寸与颜色 —— 尺寸是下游钉碰撞盒的唯一真相源，
    转换完要核一眼。

    Raises:
        FileNotFoundError: `SRC` 不存在（STEP 原件不随本包分发，见模块 docstring）。
    """
    if not SRC.is_dir():
        raise FileNotFoundError(
            f"STEP 原件目录不存在：{SRC}。它不随本包分发；"
            "产物已在 kit_assets/objects/，正常使用不需要跑本工具。"
        )
    DST.mkdir(parents=True, exist_ok=True)
    for step_path in sorted(SRC.glob("*.STEP")):
        stem = step_path.stem
        mesh = _load_merged(step_path)
        color = _color_from_name(stem)
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=color)

        # 干净的输出名：去掉 hex 后缀里的 # 与中文，保留尺寸档。
        safe = re.sub(r"#[0-9a-fA-F]{6}", "", stem).strip("_").strip()
        safe = re.sub(r"[^0-9A-Za-z_]+", "_", safe).strip("_") or stem

        visual_path = DST / f"{safe}_visual.glb"
        mesh.export(visual_path)

        collision = trimesh.convex.convex_hull(mesh)
        collision_path = DST / f"{safe}_collision.obj"
        collision.export(collision_path)

        ext = np.round(mesh.bounding_box.extents, 4).tolist()
        print(f"{step_path.name:28s} -> {visual_path.name} + {collision_path.name} "
              f"| extents(m)={ext} color={color[:3]}")


if __name__ == "__main__":
    convert_all()
