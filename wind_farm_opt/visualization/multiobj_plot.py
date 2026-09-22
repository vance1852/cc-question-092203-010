"""多目标优化可视化。

包含三类候选会评审图：

- :func:`plot_pareto_pairwise` —— 三个目标两两投影的 Pareto 散点矩阵，
  每个方案标注稳定编号，膝点方案以星号高亮；
- :func:`plot_pareto_parallel` —— 归一化平行坐标图，直观展示各方案
  在三个目标上的取舍与偏好权重；
- :func:`plot_collection_network` —— 单个方案的机位与 MST 集电网络
  俯视图，可回溯每条电缆边与升压站位置。
"""

from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

from ..constraints.boundary import SiteBoundary
from ..optimization.multiobjective import (
    MultiObjectiveResult,
    ParetoSolution,
)
from ..optimization.objectives import OBJECTIVE_KEYS, OBJECTIVE_LABELS, OBJECTIVE_DIRECTIONS
from .plotting import set_chinese_font


_PAIRS = (
    ("net_aep_mwh", "lcoe_yuan_per_kwh"),
    ("net_aep_mwh", "collection_length_m"),
    ("lcoe_yuan_per_kwh", "collection_length_m"),
)


def _solution_key_map(result: MultiObjectiveResult) -> dict[str, int]:
    return {key: i for i, key in enumerate(result.objective_keys)}


def plot_pareto_pairwise(
    result: MultiObjectiveResult,
    title: str = "Pareto 非支配解集（目标两两投影）",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """绘制三目标两两投影的 Pareto 散点矩阵。

    每个点标注其稳定方案编号（P01、P02……），膝点用红色五角星标出。
    """
    set_chinese_font()

    key_idx = _solution_key_map(result)
    solutions = result.solutions
    vectors = np.array([s.objective_vector() for s in solutions])
    knee_id = result.knee_solution.solution_id

    # 拥挤度端点为 inf，不能直接送入颜色映射（会被遮罩并产生渲染伪影），
    # 用有限最大值替代，仅影响着色，不影响数值文件。
    crowd_raw = np.array([s.crowding_distance for s in solutions], dtype=float)
    finite = crowd_raw[np.isfinite(crowd_raw)]
    crowd_max = float(finite.max()) if finite.size else 1.0
    crowd_colors = np.where(np.isfinite(crowd_raw), crowd_raw, crowd_max)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    for ax, (kx, ky) in zip(axes, _PAIRS):
        ix, iy = key_idx[kx], key_idx[ky]
        xs, ys = vectors[:, ix], vectors[:, iy]

        scatter = ax.scatter(
            xs, ys,
            c=crowd_colors,
            cmap="viridis",
            s=70,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )

        for s, x, y in zip(solutions, xs, ys):
            if s.solution_id == knee_id:
                continue
            ax.annotate(
                s.solution_id,
                (x, y),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7.5,
                color="dimgray",
                zorder=4,
            )

        knee = result.knee_solution
        kv = knee.objective_vector()
        ax.scatter(
            [kv[ix]], [kv[iy]],
            marker="*",
            s=420,
            color="red",
            edgecolors="darkred",
            linewidths=1.0,
            zorder=5,
            label=f"膝点 {knee_id}",
        )
        ax.annotate(
            knee_id,
            (kv[ix], kv[iy]),
            textcoords="offset points",
            xytext=(10, 8),
            fontsize=9,
            fontweight="bold",
            color="red",
            zorder=6,
        )

        ax.set_xlabel(OBJECTIVE_LABELS[kx])
        ax.set_ylabel(OBJECTIVE_LABELS[ky])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Pareto 散点矩阵已保存到: {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_pareto_parallel(
    result: MultiObjectiveResult,
    title: str = "Pareto 方案归一化平行坐标（上方为更优）",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """绘制归一化平行坐标图。

    三个纵轴均为 min-max 归一化值，且统一为"越靠上越优"
    （LCOE 与集电线路长度做了方向翻转）。膝点方案加粗高亮，
    轴标签上注明偏好权重。
    """
    set_chinese_font()

    solutions = result.solutions
    vectors = np.array([s.objective_vector() for s in solutions])
    directions = np.array(
        [result.objective_directions[k] for k in result.objective_keys]
    )
    benefit = vectors * directions[np.newaxis, :]

    norm = result.normalization
    normalized = np.zeros_like(benefit)
    for m, key in enumerate(result.objective_keys):
        info = norm[key]
        best = info["ideal"] * info["direction"]
        worst = info["nadir"] * info["direction"]
        span = best - worst
        normalized[:, m] = 0.5 if span <= 0 else (benefit[:, m] - worst) / span

    x = np.arange(len(result.objective_keys))
    fig, ax = plt.subplots(figsize=(11, 6))

    cmap = plt.get_cmap("tab20")
    knee_id = result.knee_solution.solution_id

    for i, s in enumerate(solutions):
        is_knee = s.solution_id == knee_id
        ax.plot(
            x, normalized[i],
            color="red" if is_knee else cmap(i % 20),
            linewidth=2.4 if is_knee else 1.0,
            alpha=1.0 if is_knee else 0.45,
            marker="*" if is_knee else "o",
            markersize=13 if is_knee else 4,
            zorder=5 if is_knee else 2,
        )

    ax.set_xticks(x)
    labels = []
    for key in result.objective_keys:
        w = norm[key]["weight"]
        arrow = "↑" if OBJECTIVE_DIRECTIONS[key] > 0 else "↓"
        labels.append(f"{OBJECTIVE_LABELS[key]}\n{arrow} 偏好权重 {w:.2f}")
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("归一化目标值（1=Pareto 最优，0=最差）")
    ax.set_ylim(-0.05, 1.12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.plot([], [], color="red", linewidth=2.4, marker="*",
            label=f"膝点方案 {knee_id}")
    ax.legend(loc="lower center", ncol=1, fontsize=10)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Pareto 平行坐标图已保存到: {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_collection_network(
    solution: ParetoSolution,
    boundary: SiteBoundary,
    rotor_diameters: np.ndarray,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """绘制单个 Pareto 方案的机位与 MST 集电网络布局。"""
    set_chinese_font()

    positions = solution.positions
    network = solution.objectives.network
    if network is None:
        raise ValueError("方案缺少集电网络明细，无法绘图")

    if title is None:
        obj = solution.objectives.as_dict()
        title = (
            f"方案 {solution.solution_id} 集电网络布局 | "
            f"净AEP {obj['net_aep_mwh']/1e3:.2f} GWh, "
            f"LCOE {obj['lcoe_yuan_per_kwh']:.3f} 元/kWh, "
            f"线路 {obj['collection_length_m']/1e3:.2f} km"
        )

    fig, ax = plt.subplots(figsize=(10, 8))

    poly = Polygon(
        boundary.vertices,
        facecolor="lightgreen",
        edgecolor="darkgreen",
        linewidth=2,
        alpha=0.25,
    )
    ax.add_patch(poly)

    sub_xy = network.substation_xy
    sub_node = network.substation_node

    for a, b, length in network.edges:
        pa = sub_xy if a == sub_node else positions[a]
        pb = sub_xy if b == sub_node else positions[b]
        is_sub = (a == sub_node or b == sub_node)
        ax.plot(
            [pa[0], pb[0]], [pa[1], pb[1]],
            color="darkorange" if is_sub else "steelblue",
            linewidth=2.0 if is_sub else 1.4,
            alpha=0.9,
            zorder=2,
        )
        mx, my = (pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0
        ax.text(
            mx, my, f"{length/1e3:.2f}km",
            fontsize=7, color="dimgray",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.7),
            zorder=3,
        )

    for i, (pos, d) in enumerate(zip(positions, rotor_diameters)):
        circle = Circle(pos, d / 2.0, facecolor="white",
                        edgecolor="navy", linewidth=1.5, zorder=4)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], str(i), ha="center", va="center",
                fontsize=8, fontweight="bold", color="navy", zorder=5)

    ax.scatter(
        [sub_xy[0]], [sub_xy[1]],
        marker="s", s=260, color="red", edgecolors="darkred",
        linewidths=1.5, zorder=6, label="升压站",
    )
    ax.text(sub_xy[0], sub_xy[1], "S", ha="center", va="center",
            fontsize=10, fontweight="bold", color="white", zorder=7)

    margin = 0.1
    x_range = boundary.x_max - boundary.x_min
    y_range = boundary.y_max - boundary.y_min
    ax.set_xlim(boundary.x_min - margin * x_range,
                boundary.x_max + margin * x_range)
    ax.set_ylim(boundary.y_min - margin * y_range,
                boundary.y_max + margin * y_range)
    ax.set_aspect("equal")
    ax.set_xlabel("X 坐标 (m)")
    ax.set_ylabel("Y 坐标 (m)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"集电网络布局图已保存到: {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_pareto_convergence(
    result: MultiObjectiveResult,
    title: str = "多目标优化存档演化",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """绘制外部存档规模与多样性间距指标随代数的变化。"""
    set_chinese_font()

    history = result.convergence_history
    if not history:
        return

    gens = [h["generation"] for h in history]
    sizes = [h["archive_size"] for h in history]
    spacing = [h["spacing"] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(gens, sizes, "b-", linewidth=2)
    ax1.set_ylabel("Pareto 存档解数")
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(gens, spacing, "g-", linewidth=2)
    ax2.set_xlabel("迭代代数")
    ax2.set_ylabel("Schott 间距指标（越小越均匀）")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"多目标存档演化图已保存到: {save_path}")

    if show:
        plt.show()
    plt.close(fig)
