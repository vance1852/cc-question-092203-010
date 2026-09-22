"""多目标 Pareto 结果可视化。

包含：

- 目标空间两两散点图（净AEP / LCOE / 集电线路长度），标注方案编号与膝点
- 归一化目标的平行坐标图，直观展示评审会上的取舍关系
- 指定方案（默认膝点）的机位与集电连通网络布局图
"""

from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

from ..constraints.boundary import SiteBoundary
from ..optimization.multi_objective import MultiObjectiveResult, ParetoSolution
from .plotting import set_chinese_font


def _objective_rows(result: MultiObjectiveResult):
    ids = [s.solution_id for s in result.solutions]
    aep = np.array([s.objectives.net_aep_gwh for s in result.solutions])
    lcoe = np.array([s.objectives.lcoe for s in result.solutions])
    cable = np.array([s.objectives.cable_length_m for s in result.solutions]) / 1e3
    return ids, aep, lcoe, cable


def plot_pareto_front(
    result: MultiObjectiveResult,
    title: str = "Pareto 非支配解集",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """绘制三个目标两两组合的散点图矩阵，膝点以红星标注。

    每个点旁标注稳定编号（P00、P01……），与结果文件中的机位数据一一对应。
    """
    set_chinese_font()

    ids, aep, lcoe, cable_km = _objective_rows(result)
    knee_id = result.knee_solution.solution_id

    panels = [
        (aep, lcoe, "净AEP (GWh/年)", "度电成本 (元/kWh)"),
        (aep, cable_km, "净AEP (GWh/年)", "集电线路长度 (km)"),
        (lcoe, cable_km, "度电成本 (元/kWh)", "集电线路长度 (km)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    scores = np.array([s.weighted_score for s in result.solutions])
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=scores.min(), vmax=scores.max())

    for ax, (xs, ys, xlabel, ylabel) in zip(axes, panels):
        order = np.argsort(-scores, kind="stable")  # 得分差的先画，膝点类后画在上层
        for idx in order:
            is_knee = ids[idx] == knee_id
            ax.scatter(
                xs[idx], ys[idx],
                s=140 if is_knee else 70,
                color="crimson" if is_knee else cmap(norm(scores[idx])),
                marker="*" if is_knee else "o",
                edgecolors="black",
                linewidths=1.2 if is_knee else 0.6,
                zorder=5 if is_knee else 3,
            )
            ax.annotate(
                ids[idx],
                (xs[idx], ys[idx]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
                fontweight="bold" if is_knee else "normal",
                color="crimson" if is_knee else "dimgray",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
    cbar.set_label("偏好加权差距（越小越接近膝点）")

    w = result.knee_weights
    fig.suptitle(
        f"{title}（膝点 {knee_id}；偏好权重 "
        f"AEP {w['aep']:.2f} / LCOE {w['lcoe']:.2f} / 集电 {w['cable']:.2f}）",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 0.97, 0.93))

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Pareto 前沿图已保存到: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_pareto_parallel(
    result: MultiObjectiveResult,
    title: str = "Pareto 方案目标取舍（归一化）",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """绘制归一化目标的平行坐标图，膝点加粗为红色。"""
    set_chinese_font()

    fig, ax = plt.subplots(figsize=(10, 6))

    dims = ["aep", "lcoe", "cable"]
    labels = ["净AEP", "度电成本", "集电线路长度"]
    x = np.arange(len(dims))
    knee_id = result.knee_solution.solution_id

    for sol in result.solutions:
        y = [sol.normalized[d] for d in dims]
        is_knee = sol.solution_id == knee_id
        ax.plot(
            x, y,
            color="crimson" if is_knee else "steelblue",
            linewidth=2.6 if is_knee else 1.0,
            alpha=1.0 if is_knee else 0.35,
            zorder=5 if is_knee else 2,
        )
        if is_knee:
            ax.scatter(x, y, color="crimson", s=70, marker="*", zorder=6)
            ax.annotate(
                sol.solution_id, (x[-1], y[-1]),
                textcoords="offset points", xytext=(8, 0),
                fontsize=10, fontweight="bold", color="crimson",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("归一化目标值（1 = 最优，0 = 最劣）")
    ax.set_ylim(-0.05, 1.12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # 在每个维度上标注原始量纲的最优/最劣值
    bounds = result.normalization_bounds
    raw = [
        f"{bounds['net_aep_gwh']['min']:.1f}~{bounds['net_aep_gwh']['max']:.1f} GWh",
        f"{bounds['lcoe']['min']:.3f}~{bounds['lcoe']['max']:.3f} 元/kWh",
        f"{bounds['cable_length_m']['min']/1e3:.2f}~{bounds['cable_length_m']['max']/1e3:.2f} km",
    ]
    for xi, text in zip(x, raw):
        ax.annotate(text, (xi, 1.08), ha="center", fontsize=8, color="dimgray")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"平行坐标图已保存到: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_pareto_convergence(
    result: MultiObjectiveResult,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """绘制多目标搜索过程中前沿规模与各目标最优值的变化。"""
    set_chinese_font()

    hist = result.history
    iters = [h["iteration"] for h in hist]
    n_front = [h["n_non_dominated"] for h in hist]
    best_aep = [h["best_aep_mwh"] / 1e3 for h in hist]
    min_cable = [h["min_cable_m"] / 1e3 for h in hist]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(iters, n_front, "b-", linewidth=2)
    axes[0].set_ylabel("非支配解数量")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iters, best_aep, "g-", linewidth=2)
    axes[1].set_ylabel("前沿最佳净AEP (GWh)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(iters, min_cable, "r-", linewidth=2)
    axes[2].set_ylabel("前沿最短线缆 (km)")
    axes[2].set_xlabel("迭代代数")
    axes[2].grid(True, alpha=0.3)

    algo_name = "NSGA-II" if result.algorithm == "nsga2" else "MOPSO"
    axes[0].set_title(
        title or f"{algo_name} 多目标搜索过程",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"多目标收敛曲线已保存到: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_solution_network(
    solution: ParetoSolution,
    boundary: SiteBoundary,
    rotor_diameters: np.ndarray,
    turbine_losses: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """绘制某个 Pareto 方案的机位布局与集电连通网络（MST）。

    实线为风机间线缆，虚线为风机到升压站的连接；升压站以红色方块标出。
    """
    set_chinese_font()

    positions = solution.positions
    network = solution.cable_network
    substation = network.substation_position

    fig, ax = plt.subplots(figsize=(10, 8))

    poly = Polygon(
        boundary.vertices,
        facecolor="lightgreen", edgecolor="darkgreen",
        linewidth=2, alpha=0.25,
    )
    ax.add_patch(poly)

    # 集电网络边
    for a, b, length in network.edges:
        if a < 0 or b < 0:
            t = max(a, b)
            p1, p2 = positions[t], substation
            ls, color, lw, alpha = "--", "darkorange", 1.6, 0.9
        else:
            p1, p2 = positions[a], positions[b]
            ls, color, lw, alpha = "-", "saddlebrown", 1.4, 0.75
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            color=color, linestyle=ls, linewidth=lw, alpha=alpha, zorder=2,
        )

    # 机位
    if turbine_losses is not None:
        norm = plt.Normalize(vmin=0, vmax=max(30.0, float(np.max(turbine_losses))))
        cmap = plt.get_cmap("YlOrRd")
        for i, (pos, d, loss) in enumerate(zip(positions, rotor_diameters, turbine_losses)):
            ax.add_patch(Circle(
                pos, d / 2.0, facecolor=cmap(norm(loss)),
                edgecolor="black", linewidth=1.2, alpha=0.9, zorder=3,
            ))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("尾流损失 (%)")
    else:
        for pos, d in zip(positions, rotor_diameters):
            ax.add_patch(Circle(
                pos, d / 2.0, facecolor="steelblue",
                edgecolor="darkblue", linewidth=1.2, alpha=0.85, zorder=3,
            ))

    # 升压站
    ax.scatter(
        [substation[0]], [substation[1]],
        marker="s", s=260, c="crimson", edgecolors="black",
        linewidths=1.5, zorder=6, label="升压站",
    )
    ax.annotate(
        "升压站", substation,
        textcoords="offset points", xytext=(10, 10),
        fontsize=10, fontweight="bold", color="crimson",
    )

    margin = 0.1
    x_range = boundary.x_max - boundary.x_min
    y_range = boundary.y_max - boundary.y_min
    ax.set_xlim(boundary.x_min - margin * x_range, boundary.x_max + margin * x_range)
    ax.set_ylim(boundary.y_min - margin * y_range, boundary.y_max + margin * y_range)
    ax.set_aspect("equal")
    ax.set_xlabel("X 坐标 (m)")
    ax.set_ylabel("Y 坐标 (m)")

    o = solution.objectives
    ax.set_title(
        title or (
            f"方案 {solution.solution_id} 机位与集电网络｜"
            f"净AEP {o.net_aep_gwh/1e3:.2f} GWh，"
            f"LCOE {o.lcoe:.3f} 元/kWh，"
            f"线缆 {o.cable_length_m/1e3:.2f} km"
        ),
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"方案网络图已保存到: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
