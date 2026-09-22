"""场内集电线路（机位-升压站连通网络）估算。

将每台风机与升压站视为无向完全图中的节点，以欧氏距离为边权，用 Prim
算法求最小生成树（MST）。MST 给出连接全部机位与升压站所需的最短线缆
长度（放射式/树状集电网络的下界估计），同时返回边列表，便于在布局图
上回溯每一段线缆的走向。
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CableNetwork:
    """集电连通网络估算结果。

    Parameters
    ----------
    total_length : float
        线缆总长度 (m)，即最小生成树所有边权之和
    substation_position : np.ndarray
        升压站坐标 (2,)
    edges : list[tuple[int, int, float]]
        树边列表，每项为 (节点a, 节点b, 长度m)。
        风机节点编号为 0 .. N-1，升压站节点编号为 -1。
    max_turbine_chain : int
        不考虑升压站时，风机之间单个连通分量的最大边数（供串联深度参考）
    """

    total_length: float
    substation_position: np.ndarray
    edges: list[tuple[int, int, float]] = field(default_factory=list)
    max_turbine_chain: int = 0

    def turbine_edges(self) -> list[tuple[int, int, float]]:
        """仅返回风机之间的边 (i, j, length)。"""
        return [(a, b, w) for a, b, w in self.edges if a >= 0 and b >= 0]

    def substation_edges(self) -> list[tuple[int, float]]:
        """返回风机直接连到升压站的边 (turbine_idx, length)。"""
        return [(max(a, b), w) for a, b, w in self.edges if a < 0 or b < 0]


def estimate_substation_position(positions: np.ndarray) -> np.ndarray:
    """未显式给定时，以机位重心（1-中位数的近似）估算升压站位置。

    Parameters
    ----------
    positions : np.ndarray
        风机位置 (N, 2)

    Returns
    -------
    np.ndarray
        升压站坐标 (2,)
    """
    positions = np.asarray(positions, dtype=np.float64)
    return positions.mean(axis=0)


def estimate_cable_network(
    positions: np.ndarray,
    substation_position: np.ndarray | None = None,
) -> CableNetwork:
    """计算机位到升压站连通网络的最小生成树与线缆总长度。

    节点 0..N-1 为风机，节点 N 为升压站。Prim 算法在相等边权处按节点
    编号做确定性抉择，保证相同布局得到完全一致的网络与长度。

    Parameters
    ----------
    positions : np.ndarray
        风机位置 (N, 2)
    substation_position : Optional[np.ndarray]
        升压站坐标 (2,)；为 None 时取机位重心

    Returns
    -------
    CableNetwork
        连通网络估算结果
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"位置数组形状应为 (N, 2)，实际为 {positions.shape}")

    n_turb = positions.shape[0]
    if n_turb < 1:
        raise ValueError("至少需要 1 台风机")

    if substation_position is None:
        substation_position = estimate_substation_position(positions)
    substation_position = np.asarray(substation_position, dtype=np.float64).reshape(2)

    # 节点 0..n_turb-1 风机，节点 n_turb 升压站
    nodes = np.vstack([positions, substation_position[np.newaxis, :]])
    n_nodes = n_turb + 1
    sub_idx = n_turb

    # ---- Prim 最小生成树（稠密图，O(V^2)）----
    in_tree = np.zeros(n_nodes, dtype=bool)
    best_dist = np.full(n_nodes, np.inf, dtype=np.float64)
    best_parent = np.full(n_nodes, -1, dtype=int)

    # 从升压站开始生长，边列表在同长度下仍可稳定复现
    best_dist[sub_idx] = 0.0
    edges: list[tuple[int, int, float]] = []
    total_length = 0.0

    for _ in range(n_nodes):
        # 选取树外距离最小的节点；平局取编号最小者（确定性）
        candidates = np.where(~in_tree, best_dist, np.inf)
        u = int(np.argmin(candidates))
        if not np.isfinite(candidates[u]):
            break  # 完全图理论上不会发生
        in_tree[u] = True
        if best_parent[u] >= 0:
            p = int(best_parent[u])
            w = float(best_dist[u])
            # 统一边的存储方向：小编号在前，升压站用 -1 标记，便于回溯
            a, b = (u, p) if u < p else (p, u)
            if a == sub_idx:
                a = -1
            if b == sub_idx:
                b = -1
            edges.append((a, b, w))
            total_length += w

        diff = nodes - nodes[u]
        dists = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        # 严格更小才更新；相等时保留先建立的父节点 → 结果确定
        update = (~in_tree) & (dists < best_dist)
        best_dist[update] = dists[update]
        best_parent[update] = u

    # 风机间接入深度（从某棵风机子树出发，不经过升压站的最大串联台数）
    max_chain = _max_component_edges(n_turb, edges)

    return CableNetwork(
        total_length=float(total_length),
        substation_position=substation_position.copy(),
        edges=edges,
        max_turbine_chain=max_chain,
    )


def _max_component_edges(
    n_turb: int,
    edges: list[tuple[int, int, float]],
) -> int:
    """去掉升压站后，单个风机连通分量中的最大边数。"""
    adjacency: dict[int, list[int]] = {i: [] for i in range(n_turb)}
    for a, b, _w in edges:
        if a >= 0 and b >= 0:
            adjacency[a].append(b)
            adjacency[b].append(a)

    visited = set()
    max_edges = 0
    for start in range(n_turb):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        edge_count = 0
        while stack:
            node = stack.pop()
            for nxt in adjacency[node]:
                edge_count += 1
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        # 无向边被计数两次
        max_edges = max(max_edges, edge_count // 2)
    return max_edges
