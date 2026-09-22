"""多目标机位布局优化（NSGA-II）。

同时优化三个目标：净AEP（最大）、度电成本LCOE（最小）、集电线路
总长度（最小）。算法采用带精英保留的 NSGA-II：

- 快速非支配排序 + 拥挤度维持 Pareto 解集的多样性；
- 外部存档在每代后并入种群、重新做非支配排序并按拥挤度裁剪；
- 所有个体均经边界/间距修复，存档只接受满足约束的可行解；
- 结果排序、编号、膝点选择均为确定性操作，相同种子重复运行
  得到完全一致的 Pareto 顺序。
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..constraints.boundary import SiteBoundary
from ..constraints.spacing import (
    check_min_spacing,
    compute_min_spacing_from_diameters,
    enforce_min_spacing,
)
from .objectives import (
    OBJECTIVE_DIRECTIONS,
    OBJECTIVE_KEYS,
    OBJECTIVE_LABELS,
    LayoutObjectives,
    MultiObjectiveEvaluator,
)
from .baseline import generate_grid_layout


@dataclass
class MultiObjectiveConfig:
    """多目标优化配置。

    Parameters
    ----------
    population_size : int
        种群大小
    max_generations : int
        最大迭代代数
    crossover_rate : float
        均匀交叉概率
    mutation_rate : float
        单基因变异概率
    mutation_strength : float
        变异强度（坐标标准差占场地范围的比例）
    archive_size : int
        外部 Pareto 存档容量上限
    min_spacing_multiple : float
        最小间距倍数（转子直径倍数）
    seed : Optional[int]
        随机种子；相同种子保证可复现的排序与编号
    preference_weights : dict[str, float]
        膝点方案的归一化偏好权重（无需归一，内部自动归一化），
        键为 ``net_aep_mwh`` / ``lcoe_yuan_per_kwh`` / ``collection_length_m``
    """

    population_size: int = 50
    max_generations: int = 80
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    mutation_strength: float = 0.08
    archive_size: int = 80
    min_spacing_multiple: float = 5.0
    seed: Optional[int] = 42
    preference_weights: dict[str, float] = field(default_factory=lambda: {
        "net_aep_mwh": 1.0 / 3.0,
        "lcoe_yuan_per_kwh": 1.0 / 3.0,
        "collection_length_m": 1.0 / 3.0,
    })


@dataclass
class ParetoSolution:
    """Pareto 解集中的单个方案（机位 + 目标原值 + 网络明细）。"""

    solution_id: str
    positions: np.ndarray
    objectives: LayoutObjectives
    rank: int = 0
    crowding_distance: float = 0.0

    def objective_vector(self) -> np.ndarray:
        """三个目标的原始量纲向量，顺序见 :data:`OBJECTIVE_KEYS`。"""
        d = self.objectives.as_dict()
        return np.array([d[k] for k in OBJECTIVE_KEYS], dtype=np.float64)

    def to_dict(self) -> dict:
        """序列化为可写入 JSON 的字典（含机位与集电网络，可完整回溯）。"""
        net = self.objectives.network
        edges = []
        if net is not None:
            for a, b, length in net.edges:
                edges.append({
                    "node_i": int(a),
                    "node_j": int(b),
                    "length_m": float(length),
                    "connects_substation": int(b) == net.substation_node,
                })

        return {
            "solution_id": self.solution_id,
            "positions": self.positions.tolist(),
            "objectives": {
                key: float(self.objective_vector()[i])
                for i, key in enumerate(OBJECTIVE_KEYS)
            },
            "objective_units": {
                "net_aep_mwh": "MWh/year",
                "lcoe_yuan_per_kwh": "CNY/kWh",
                "collection_length_m": "m",
            },
            "collection_cost_wanyuan": float(self.objectives.collection_cost_wanyuan),
            "total_capital_cost_wanyuan": float(
                self.objectives.total_capital_cost_wanyuan
            ),
            # 端点拥挤度为无穷大，序列化为 null 以保持 JSON 标准合法。
            "crowding_distance": (
                float(self.crowding_distance)
                if np.isfinite(self.crowding_distance)
                else None
            ),
            "substation_xy": (
                net.substation_xy.tolist() if net is not None else None
            ),
            "collection_network_edges": edges,
        }


@dataclass
class MultiObjectiveResult:
    """多目标优化结果。"""

    solutions: list[ParetoSolution]
    knee_solution: ParetoSolution
    objective_keys: tuple[str, ...]
    objective_directions: dict[str, int]
    objective_labels: dict[str, str]
    preference_weights: dict[str, float]
    normalization: dict[str, dict[str, float]]
    algorithm: str
    seed: Optional[int]
    config: MultiObjectiveConfig
    convergence_history: list[dict]
    substation_xy: np.ndarray

    @property
    def knee_index(self) -> int:
        """膝点方案在 ``solutions`` 中的索引。"""
        return next(
            i for i, s in enumerate(self.solutions)
            if s.solution_id == self.knee_solution.solution_id
        )


def fast_non_dominated_sort(objective_matrix: np.ndarray) -> list[list[int]]:
    """快速非支配排序。

    Parameters
    ----------
    objective_matrix : np.ndarray
        目标矩阵 (M, K)，约定所有目标均为"越小越好"

    Returns
    -------
    list[list[int]]
        逐层非支配前沿，每个前沿是个体索引列表（按索引升序，保证确定性）
    """
    n = objective_matrix.shape[0]
    domination_sets: list[set[int]] = [set() for _ in range(n)]
    domination_counts = np.zeros(n, dtype=int)

    for p in range(n):
        for q in range(p + 1, n):
            le_p = np.all(objective_matrix[p] <= objective_matrix[q])
            le_q = np.all(objective_matrix[q] <= objective_matrix[p])
            lt_p = np.any(objective_matrix[p] < objective_matrix[q])
            lt_q = np.any(objective_matrix[q] < objective_matrix[p])

            if le_p and lt_p:
                domination_sets[p].add(q)
                domination_counts[q] += 1
            elif le_q and lt_q:
                domination_sets[q].add(p)
                domination_counts[p] += 1

    fronts: list[list[int]] = []
    current = [i for i in range(n) if domination_counts[i] == 0]
    visited = 0

    while current:
        fronts.append(sorted(current))
        visited += len(current)
        next_front: set[int] = set()
        for p in current:
            for q in domination_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.add(q)
        current = list(next_front)

    return fronts


def crowding_distance(objective_matrix: np.ndarray) -> np.ndarray:
    """计算一个前沿内各解的拥挤度距离。

    目标方向无需提前翻转：传入的矩阵应已经统一为"越小越好"，
    各维度端点均赋予无穷大拥挤度。
    """
    n, k = objective_matrix.shape
    distances = np.zeros(n, dtype=np.float64)
    if n <= 2:
        distances[:] = np.inf
        return distances

    indices = np.arange(n)
    for m in range(k):
        order = np.lexsort((indices, objective_matrix[:, m]))
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        col_min = objective_matrix[order[0], m]
        col_max = objective_matrix[order[-1], m]
        span = col_max - col_min
        if span <= 0.0:
            continue
        for r in range(1, n - 1):
            if np.isfinite(distances[order[r]]):
                distances[order[r]] += (
                    objective_matrix[order[r + 1], m]
                    - objective_matrix[order[r - 1], m]
                ) / span

    return distances


def _canonical_sort_key(vector: np.ndarray):
    """Pareto 方案的确定性排序键：AEP 降序、LCOE 升序、线路长度升序。"""
    return (-vector[0], vector[1], vector[2])


class MultiObjectiveOptimizer:
    """NSGA-II 机位布局多目标优化器。

    支持以 GA 或 PSO 风格的算子驱动（PSO 风格采用相同的变异/交叉框架，
    但保留基于粒子群的初始扩散与惯性扰动），默认 ``algorithm="nsga2"``。
    """

    def __init__(
        self,
        n_turbines: int,
        rotor_diameters: np.ndarray,
        boundary: SiteBoundary,
        evaluator: MultiObjectiveEvaluator,
        config: Optional[MultiObjectiveConfig] = None,
        algorithm: str = "nsga2",
    ) -> None:
        self.n_turbines = n_turbines
        self.rotor_diameters = np.asarray(rotor_diameters, dtype=np.float64)
        self.boundary = boundary
        self.evaluator = evaluator
        self.config = config if config is not None else MultiObjectiveConfig()
        self.algorithm = algorithm.lower()

        self.rng = np.random.default_rng(self.config.seed)
        # 初始网格注入使用独立派生流，避免改变随机序列的整体结构。
        self._grid_rng = np.random.default_rng(
            None if self.config.seed is None else self.config.seed + 10_007
        )

        self.min_spacing = compute_min_spacing_from_diameters(
            self.rotor_diameters, self.config.min_spacing_multiple
        )

        self.n_dim = n_turbines * 2
        self.x_range = boundary.x_max - boundary.x_min
        self.y_range = boundary.y_max - boundary.y_min

        # 目标值缓存：同一布局（毫米级取整）不重复评估，
        # 键只含坐标，因此缓存本身与访问顺序无关，保持确定性。
        self._eval_cache: dict[bytes, LayoutObjectives] = {}

    # ------------------------------------------------------------------
    # 布局生成与修复
    # ------------------------------------------------------------------

    def _generate_valid_layout(self) -> np.ndarray:
        """生成一个满足边界与间距约束的初始布局。"""
        for _ in range(100):
            try:
                positions = self.boundary.sample_random_points(
                    self.n_turbines, self.rng, max_attempts=50
                )
                valid, _ = check_min_spacing(positions, self.min_spacing)
                if valid:
                    return positions
            except RuntimeError:
                pass

            try:
                positions = self.boundary.sample_random_points(
                    self.n_turbines, self.rng, max_attempts=50
                )
                positions = enforce_min_spacing(
                    positions, self.min_spacing, self.boundary, self.rng
                )
                return positions
            except RuntimeError:
                continue

        raise RuntimeError("无法生成满足约束的初始布局")

    def _is_feasible(self, positions: np.ndarray) -> bool:
        inside = self.boundary.contains_all(positions)
        valid, _ = check_min_spacing(positions, self.min_spacing)
        return bool(inside.all() and valid)

    def _repair(self, individual: np.ndarray) -> np.ndarray:
        """修复违反边界/间距约束的个体。"""
        positions = individual.reshape(self.n_turbines, 2)

        for i in range(self.n_turbines):
            if not self.boundary.contains_point(positions[i]):
                positions[i] = self.boundary.project_to_boundary(positions[i])

        valid, _ = check_min_spacing(positions, self.min_spacing)
        inside = self.boundary.contains_all(positions).all()

        if not (valid and inside):
            try:
                positions = enforce_min_spacing(
                    positions, self.min_spacing, self.boundary, self.rng
                )
            except RuntimeError:
                pass

        return positions.flatten()

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------

    def _evaluate_one(
        self,
        flat: np.ndarray,
        retain_network: bool,
    ) -> Optional[LayoutObjectives]:
        """评估单个个体；不可行或计算失败返回 None。"""
        positions = flat.reshape(self.n_turbines, 2)
        if not self._is_feasible(positions):
            return None

        key = np.round(positions, decimals=3).tobytes()
        cached = self._eval_cache.get(key)
        if cached is not None:
            if retain_network and cached.network is None:
                cached = self.evaluator.evaluate(positions, retain_network=True)
                self._eval_cache[key] = cached
            return cached

        try:
            obj = self.evaluator.evaluate(positions, retain_network=retain_network)
        except Exception:
            return None

        self._eval_cache[key] = obj
        return obj

    def _evaluate_population(
        self,
        population: np.ndarray,
        retain_network: bool = False,
    ) -> tuple[np.ndarray, list[Optional[LayoutObjectives]]]:
        """返回统一为"越小越好"的目标矩阵与逐个体目标对象。

        不可行个体以 NaN 行标记，由调用方决定替换策略。
        """
        pop_size = population.shape[0]
        matrix = np.full((pop_size, len(OBJECTIVE_KEYS)), np.nan)
        results: list[Optional[LayoutObjectives]] = []

        for i in range(pop_size):
            obj = self._evaluate_one(population[i], retain_network)
            results.append(obj)
            if obj is not None:
                d = obj.as_dict()
                matrix[i] = [
                    OBJECTIVE_DIRECTIONS[k] * d[k] for k in OBJECTIVE_KEYS
                ]

        return matrix, results

    # ------------------------------------------------------------------
    # 遗传算子
    # ------------------------------------------------------------------

    def _tournament(
        self,
        population: np.ndarray,
        ranks: np.ndarray,
        crowding: np.ndarray,
    ) -> np.ndarray:
        """二元锦标赛：层级小者胜，同层级拥挤度大者胜。"""
        pop_size = population.shape[0]
        a, b = self.rng.integers(0, pop_size, size=2)
        if ranks[a] < ranks[b]:
            return population[a].copy()
        if ranks[b] < ranks[a]:
            return population[b].copy()
        return population[a].copy() if crowding[a] >= crowding[b] else population[b].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        if self.rng.random() > self.config.crossover_rate:
            return p1.copy()
        mask = self.rng.integers(0, 2, size=self.n_dim, dtype=bool)
        return np.where(mask, p1, p2)

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        mutated = individual.copy()
        for i in range(self.n_dim):
            if self.rng.random() < self.config.mutation_rate:
                span = self.x_range if i % 2 == 0 else self.y_range
                mutated[i] += self.rng.normal(
                    0.0, span * self.config.mutation_strength
                )
        return mutated

    def _make_offspring(
        self,
        population: np.ndarray,
        ranks: np.ndarray,
        crowding: np.ndarray,
    ) -> np.ndarray:
        pop_size = population.shape[0]
        offspring = np.zeros_like(population)
        for i in range(pop_size):
            p1 = self._tournament(population, ranks, crowding)
            p2 = self._tournament(population, ranks, crowding)
            child = self._crossover(p1, p2)
            child = self._mutate(child)
            offspring[i] = self._repair(child)
        return offspring

    # ------------------------------------------------------------------
    # 存档管理
    # ------------------------------------------------------------------

    def _update_archive(
        self,
        archive_flat: list[np.ndarray],
        archive_obj: list[LayoutObjectives],
        population: np.ndarray,
        objectives: list[Optional[LayoutObjectives]],
    ) -> tuple[list[np.ndarray], list[LayoutObjectives]]:
        """并入可行个体，保留非支配层并按拥挤度裁剪到存档容量。"""
        flats = list(archive_flat)
        objs = list(archive_obj)

        for flat, obj in zip(population, objectives):
            if obj is not None:
                flats.append(flat.copy())
                objs.append(obj)

        matrix = self._as_minimize_matrix(objs)
        fronts = fast_non_dominated_sort(matrix)

        # 存档始终只保留当前非支配层（第 0 层）；旧存档中被新解
        # 支配的成员在此自然淘汰。
        pareto = fronts[0]
        if len(pareto) <= self.config.archive_size:
            kept_idx = list(pareto)
        else:
            kept_idx = self._select_by_crowding(
                matrix, list(pareto), self.config.archive_size
            )

        return [flats[i] for i in kept_idx], [objs[i] for i in kept_idx]

    @staticmethod
    def _as_minimize_matrix(objs: list[LayoutObjectives]) -> np.ndarray:
        rows = []
        for o in objs:
            d = o.as_dict()
            rows.append([OBJECTIVE_DIRECTIONS[k] * d[k] for k in OBJECTIVE_KEYS])
        return np.array(rows, dtype=np.float64)

    @staticmethod
    def _select_by_crowding(
        matrix: np.ndarray,
        front: list[int],
        n_select: int,
    ) -> list[int]:
        """从前沿中按拥挤度逐个挑选（每移除一个候选即重算，保多样性）。"""
        chosen: list[int] = []
        pool = list(front)

        while len(chosen) < n_select and pool:
            sub = matrix[pool]
            distances = crowding_distance(sub)
            # 确定性次序：拥挤度降序，平手时按索引升序。
            order = sorted(
                range(len(pool)),
                key=lambda r: (-distances[r], pool[r]),
            )
            pick = order[0]
            chosen.append(pool.pop(pick))

        return chosen

    @staticmethod
    def _spacing_metric(matrix: np.ndarray) -> float:
        """Schott 间距指标：相邻解最短距离的标准差，越小分布越均匀。"""
        n = matrix.shape[0]
        if n < 3:
            return 0.0
        mins = np.full(n, np.inf)
        for i in range(n):
            diffs = np.abs(matrix - matrix[i])
            dists = np.sum(diffs, axis=1)
            dists[i] = np.inf
            mins[i] = np.min(dists)
        mean = np.mean(mins)
        return float(np.sqrt(np.mean((mins - mean) ** 2)))

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def optimize(self, verbose: bool = True) -> MultiObjectiveResult:
        """执行 NSGA-II 多目标优化。"""
        pop_size = self.config.population_size
        max_gen = self.config.max_generations

        if verbose:
            print("\n=== NSGA-II 多目标优化开始 ===")
            print(f"目标: 净AEP↑  LCOE↓  集电线路长度↓")
            print(f"风机台数: {self.n_turbines}")
            print(f"种群大小: {pop_size}")
            print(f"最大代数: {max_gen}")
            print(f"存档容量: {self.config.archive_size}")
            print(f"最小间距: {self.min_spacing:.1f} m")
            print("=" * 40)

        # 初始种群：一个规则网格方案 + 随机可行方案。
        population = np.zeros((pop_size, self.n_dim), dtype=np.float64)
        try:
            grid = generate_grid_layout(
                boundary=self.boundary,
                n_turbines=self.n_turbines,
                rotor_diameters=self.rotor_diameters,
                min_multiple=self.config.min_spacing_multiple,
                rng=self._grid_rng,
            )
            population[0] = grid.flatten()
        except Exception:
            population[0] = self._generate_valid_layout().flatten()

        for i in range(1, pop_size):
            population[i] = self._generate_valid_layout().flatten()

        matrix, obj_list = self._evaluate_population(population)
        if not self._replace_infeasible(population, matrix):
            raise RuntimeError("初始种群存在无法消除的不可行解，请检查场地与间距约束")
        matrix, obj_list = self._evaluate_population(population)

        archive_flat: list[np.ndarray] = []
        archive_obj: list[LayoutObjectives] = []
        archive_flat, archive_obj = self._update_archive(
            archive_flat, archive_obj, population, obj_list
        )

        history: list[dict] = []

        for gen in range(max_gen):
            fronts = fast_non_dominated_sort(matrix)
            ranks = np.zeros(pop_size, dtype=int)
            crowding = np.zeros(pop_size, dtype=np.float64)
            for level, front in enumerate(fronts):
                sub_dist = crowding_distance(matrix[front])
                for local_i, idx in enumerate(front):
                    ranks[idx] = level
                    crowding[idx] = sub_dist[local_i]

            offspring = self._make_offspring(population, ranks, crowding)
            off_matrix, off_obj = self._evaluate_population(offspring)
            if not self._replace_infeasible(offspring, off_matrix):
                raise RuntimeError("子代存在无法消除的不可行解，请检查场地与间距约束")
            off_matrix, off_obj = self._evaluate_population(offspring)

            # 精英合并选择：父代 + 子代共同竞争，目标对象随行保留。
            combined = np.vstack([population, offspring])
            combined_matrix = np.vstack([matrix, off_matrix])
            combined_objs: list[Optional[LayoutObjectives]] = obj_list + off_obj
            combined_fronts = fast_non_dominated_sort(combined_matrix)

            chosen: list[int] = []
            for front in combined_fronts:
                if len(chosen) + len(front) <= pop_size:
                    chosen.extend(front)
                else:
                    remaining = pop_size - len(chosen)
                    chosen.extend(
                        self._select_by_crowding(combined_matrix, front, remaining)
                    )
                    break

            population = combined[chosen]
            matrix = combined_matrix[chosen]
            obj_list = [combined_objs[i] for i in chosen]

            archive_flat, archive_obj = self._update_archive(
                archive_flat, archive_obj, population, obj_list
            )

            arch_matrix = self._as_minimize_matrix(archive_obj)
            history.append({
                "generation": gen + 1,
                "archive_size": len(archive_obj),
                "spacing": self._spacing_metric(arch_matrix),
                "best_aep_mwh": float(np.max([o.net_aep_mwh for o in archive_obj])),
                "best_lcoe": float(np.min([o.lcoe_yuan_per_kwh for o in archive_obj])),
                "best_length_m": float(np.min([o.collection_length_m for o in archive_obj])),
            })

            if verbose and (gen % 5 == 0 or gen == max_gen - 1):
                print(
                    f"Gen {gen+1:3d} | 存档: {len(archive_obj):3d} 个非支配解 | "
                    f"AEP≥{history[-1]['best_aep_mwh']/1e3:6.2f} GWh | "
                    f"LCOE≤{history[-1]['best_lcoe']:.3f} | "
                    f"线路≤{history[-1]['best_length_m']/1e3:6.2f} km"
                )

        # 最终存档：重新评估并保留完整网络明细，用于结果回溯与绘图。
        solutions = self._build_final_solutions(archive_flat)

        knee, normalization = select_knee_solution(
            solutions, self.config.preference_weights
        )

        if verbose:
            print("=" * 40)
            print(f"优化完成! Pareto 非支配解: {len(solutions)} 个")
            kd = knee.objectives.as_dict()
            print(
                f"膝点方案 {knee.solution_id}: "
                f"AEP={kd['net_aep_mwh']/1e3:.2f} GWh, "
                f"LCOE={kd['lcoe_yuan_per_kwh']:.3f} 元/kWh, "
                f"线路={kd['collection_length_m']/1e3:.2f} km"
            )

        return MultiObjectiveResult(
            solutions=solutions,
            knee_solution=knee,
            objective_keys=OBJECTIVE_KEYS,
            objective_directions=dict(OBJECTIVE_DIRECTIONS),
            objective_labels=dict(OBJECTIVE_LABELS),
            preference_weights=dict(self.config.preference_weights),
            normalization=normalization,
            algorithm=self.algorithm,
            seed=self.config.seed,
            config=self.config,
            convergence_history=history,
            substation_xy=self.evaluator.substation_xy.copy(),
        )

    def _replace_infeasible(
        self,
        population: np.ndarray,
        matrix: np.ndarray,
        max_attempts: int = 20,
    ) -> bool:
        """用全新可行布局原地替换不可行个体，直至全部可评估。

        返回是否全部替换成功；超过尝试次数仍失败的调用方应中止，
        避免 NaN 进入非支配排序。
        """
        for _ in range(max_attempts):
            bad = [i for i in range(population.shape[0]) if np.any(np.isnan(matrix[i]))]
            if not bad:
                return True
            for i in bad:
                population[i] = self._generate_valid_layout().flatten()
            fill = np.full((len(bad), len(OBJECTIVE_KEYS)), np.nan)
            for slot, i in enumerate(bad):
                obj = self._evaluate_one(population[i], retain_network=False)
                if obj is not None:
                    d = obj.as_dict()
                    fill[slot] = [OBJECTIVE_DIRECTIONS[k] * d[k] for k in OBJECTIVE_KEYS]
            matrix[bad] = fill
        return not np.any(np.isnan(matrix))

    def _build_final_solutions(
        self,
        archive_flat: list[np.ndarray],
    ) -> list[ParetoSolution]:
        """对存档重新评估、去重、确定性排序并编号。"""
        seen: set[bytes] = set()
        flats: list[np.ndarray] = []
        objs: list[LayoutObjectives] = []

        for flat in archive_flat:
            positions = flat.reshape(self.n_turbines, 2)
            key = np.round(positions, decimals=3).tobytes()
            if key in seen:
                continue
            seen.add(key)
            obj = self.evaluator.evaluate(positions, retain_network=True)
            flats.append(flat.copy())
            objs.append(obj)

        matrix = self._as_minimize_matrix(objs)
        fronts = fast_non_dominated_sort(matrix)
        pareto_idx = fronts[0]

        order = sorted(
            pareto_idx,
            key=lambda i: _canonical_sort_key(matrix[i]),
        )

        solutions: list[ParetoSolution] = []
        for seq, i in enumerate(order):
            sid = f"P{seq + 1:02d}"
            solutions.append(ParetoSolution(
                solution_id=sid,
                positions=flats[i].reshape(self.n_turbines, 2).copy(),
                objectives=objs[i],
                rank=0,
            ))

        # 拥挤度（使用原始目标方向无关的最小化矩阵），写入方案便于回溯。
        pareto_matrix = np.array([
            [OBJECTIVE_DIRECTIONS[k] * s.objectives.as_dict()[k] for k in OBJECTIVE_KEYS]
            for s in solutions
        ])
        # solutions 已按 canonical 排序，重算其在该顺序下的拥挤度。
        dist = crowding_distance(pareto_matrix)
        for s, d in zip(solutions, dist):
            s.crowding_distance = float(d)

        return solutions


def select_knee_solution(
    solutions: list[ParetoSolution],
    preference_weights: dict[str, float],
) -> tuple[ParetoSolution, dict[str, dict[str, float]]]:
    """按明确的归一化偏好从 Pareto 解集选出膝点方案。

    每个目标先按方向统一为"越大越好"，再在 Pareto 集观测范围内做
    min-max 归一化；加权效用最高者为膝点。权重自动归一化，平手时
    采用与解集相同的确定性次序（AEP 降序、LCOE 升序、长度升序）。

    Returns
    -------
    tuple[ParetoSolution, dict]
        膝点方案与各目标的归一化区间（理想点/最差点，原始量纲）
    """
    n = len(solutions)
    if n == 0:
        raise ValueError("Pareto 解集为空，无法选择膝点方案")

    vectors = np.array([s.objective_vector() for s in solutions])
    directions = np.array(
        [OBJECTIVE_DIRECTIONS[k] for k in OBJECTIVE_KEYS], dtype=np.float64
    )
    # 统一为越大越好。
    benefit = vectors * directions[np.newaxis, :]

    weights = np.array(
        [float(preference_weights.get(k, 1.0)) for k in OBJECTIVE_KEYS],
        dtype=np.float64,
    )
    if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise ValueError(
            "偏好权重必须为非负数且总和为正: "
            f"{dict(zip(OBJECTIVE_KEYS, weights))}"
        )
    weights = weights / np.sum(weights)

    normalization: dict[str, dict[str, float]] = {}
    normalized = np.zeros_like(benefit)
    for m, key in enumerate(OBJECTIVE_KEYS):
        col = benefit[:, m]
        worst = float(np.min(col))
        best = float(np.max(col))
        # 原始量纲下的理想/最差点。
        normalization[key] = {
            "ideal": best / directions[m],
            "nadir": worst / directions[m],
            "weight": float(weights[m]),
            "direction": int(directions[m]),
        }
        span = best - worst
        normalized[:, m] = 0.5 if span <= 0.0 else (col - worst) / span

    utilities = normalized @ weights

    # 取效用最大者；多个完全相等时取 canonical 次序最前者，保证稳定。
    max_util = float(np.max(utilities))
    tied = [i for i in range(n) if abs(utilities[i] - max_util) <= 1e-12]
    best_idx = sorted(tied, key=lambda i: _canonical_sort_key(vectors[i]))[0]

    normalization["_utility"] = {
        "method": "minmax_weighted_sum",
        "knee_solution_id": solutions[best_idx].solution_id,
        "knee_utility": float(max_util),
    }

    return solutions[best_idx], normalization
