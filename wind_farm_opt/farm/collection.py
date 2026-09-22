"""场内集电系统网络估算与造价模型。

集电线路长度采用"机位节点 + 升压站节点"完全图上的最小生成树(MST)估算：
在所有机位都必须接入同一座升压站、且允许线路串接的简化假设下，MST 给出
连通全场的最短电缆总长度，可用于布局方案之间的横向比较。
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CollectionNetwork:
    """集电网络估算结果。

    Parameters
    ----------
    substation_xy : np.ndarray
        升压站坐标，形状 (2,)
    edges : list[tuple[int, int, float]]
        MST 边列表 ``(节点i, 节点j, 长度m)``。机位节点编号为 0..N-1，
        升压站节点编号为 N（见 ``substation_node``）。
    total_length_m : float
        集电线路总长度 (m)
    substation_node : int
        升压站在节点表中的索引
    """

    substation_xy: np.ndarray
    edges: list[tuple[int, int, float]]
    total_length_m: float
    substation_node: int

    def turbine_edges(self) -> list[tuple[int, int, float]]:
        """仅返回机位之间的连接边。"""
        return [e for e in self.edges if e[1] != self.substation_node]

    def substation_edges(self) -> list[tuple[int, int, float]]:
        """返回直接接入升压站的边。"""
        return [e for e in self.edges if e[1] == self.substation_node]


@dataclass
class CollectionCostModel:
    """集电系统造价模型。

    Parameters
    ----------
    cable_cost_per_km_wanyuan : float
        单位长度集电线路造价 (万元/km)，综合电缆、敷设与杆塔
    switchgear_cost_per_turbine_wanyuan : float
        单台机位箱变/开关间隔等固定费用 (万元/台)，各布局相同，
        仅影响 LCOE 绝对值而不影响方案排序
    """

    cable_cost_per_km_wanyuan: float = 35.0
    switchgear_cost_per_turbine_wanyuan: float = 20.0


def estimate_collection_network(
    positions: np.ndarray,
    substation_xy: np.ndarray,
) -> CollectionNetwork:
    """用最小生成树估算机位到升压站的集电线路网络。

    使用 Prim 算法在完全欧氏图上求 MST，复杂度 O((N+1)^2)。

    Parameters
    ----------
    positions : np.ndarray
        风机位置 (N, 2)
    substation_xy : np.ndarray
        升压站位置 (2,)

    Returns
    -------
    CollectionNetwork
        集电网络（边与总长度）
    """
    positions = np.asarray(positions, dtype=np.float64)
    substation_xy = np.asarray(substation_xy, dtype=np.float64).reshape(2)

    n_turb = positions.shape[0]
    sub_node = n_turb
    nodes = np.vstack([positions, substation_xy[np.newaxis, :]])
    n_nodes = n_turb + 1

    # Prim 算法：in_tree 标记已入树节点，key 为到当前树的最短距离。
    in_tree = np.zeros(n_nodes, dtype=bool)
    key = np.full(n_nodes, np.inf, dtype=np.float64)
    parent = np.full(n_nodes, -1, dtype=int)

    # 从升压站开始生长，保证网络以升压站为根。
    key[sub_node] = 0.0
    edges: list[tuple[int, int, float]] = []
    total_length = 0.0

    for _ in range(n_nodes):
        candidates = np.where(~in_tree, key, np.inf)
        u = int(np.argmin(candidates))
        if not np.isfinite(key[u]):
            break
        in_tree[u] = True
        if parent[u] >= 0:
            length = float(np.linalg.norm(nodes[u] - nodes[parent[u]]))
            a, b = sorted((int(u), int(parent[u])))
            edges.append((a, b, length))
            total_length += length

        diff = nodes - nodes[u]
        dist = np.linalg.norm(diff, axis=1)
        update = (~in_tree) & (dist < key)
        key[update] = dist[update]
        parent[update] = u

    return CollectionNetwork(
        substation_xy=substation_xy.copy(),
        edges=edges,
        total_length_m=float(total_length),
        substation_node=sub_node,
    )


def collection_capital_cost(
    network: CollectionNetwork,
    n_turbines: int,
    model: CollectionCostModel,
) -> tuple[float, dict[str, float]]:
    """计算集电系统初始投资。

    Returns
    -------
    tuple[float, dict[str, float]]
        总投资 (万元) 与成本分项
    """
    cable_cost = network.total_length_m / 1000.0 * model.cable_cost_per_km_wanyuan
    switchgear_cost = n_turbines * model.switchgear_cost_per_turbine_wanyuan
    total = cable_cost + switchgear_cost
    return total, {
        "集电线路": float(cable_cost),
        "箱变与开关间隔": float(switchgear_cost),
    }
