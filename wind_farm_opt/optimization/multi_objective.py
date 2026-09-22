"""多目标布局优化。

同时优化三个目标：

1. 净年发电量 ``net_aep_gwh``（GWh/年，越大越好）
2. 度电成本 ``lcoe``（元/kWh，越小越好；投资中计入集电线路造价）
3. 集电线路长度 ``cable_length_m``（m，越小越好；机位-升压站 MST 估算）

所有解始终满足场地边界与最小间距约束（约束支配原则：可行解恒优于不可
行解，归档集中只保留可行解）。提供两种驱动算法：

- ``nsga2``：带精英保留、非支配排序与拥挤距离的多目标遗传算法
- ``mopso``：基于非支配归档与拥挤距离领导选择的多目标粒子群

输出的 Pareto 解集按确定规则排序编号（P00、P01……），相同种子可稳定
复现；膝点（折中）方案按显式给定的归一化权重选出。
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..constraints.boundary import SiteBoundary
from ..constraints.spacing import (
    check_min_spacing,
    compute_min_spacing_from_diameters,
    enforce_min_spacing,
)
from ..economy.cable import CableNetwork, estimate_cable_network


# 目标方向：+1 表示越大越好，-1 表示越小越好
OBJECTIVE_SPECS = (
    ("net_aep_mwh", "净AEP", "MWh/年", 1),
    ("lcoe", "度电成本", "元/kWh", -1),
    ("cable_length_m", "集电线路长度", "m", -1),
)


@dataclass(frozen=True)
class ObjectiveSet:
    """单个布局的三个目标值（原始量纲）。"""

    net_aep_mwh: float
    lcoe: float
    cable_length_m: float

    @property
    def net_aep_gwh(self) -> float:
        """净年发电量 (GWh/年)。"""
        return self.net_aep_mwh / 1e3

    def as_array(self) -> np.ndarray:
        """返回最小化形式向量 (-净AEP[MWh], LCOE, 线缆长度[m])。"""
        return np.array(
            [-self.net_aep_mwh, self.lcoe, self.cable_length_m],
            dtype=np.float64,
        )


@dataclass
class ParetoSolution:
    """Pareto 非支配解。

    Parameters
    ----------
    solution_id : str
        稳定编号，如 "P03"
    positions : np.ndarray
        机位坐标 (N, 2)
    objectives : ObjectiveSet
        三个目标的原始量纲值
    cable_network : CableNetwork
        集电连通网络（含可回溯的树边）
    crowding_distance : float
        拥挤距离（越大表示该区域越稀疏）
    normalized : dict[str, float]
        各目标归一化到 [0, 1] 的值（1 表示该目标上最好）
    weighted_score : float
        按偏好权重计算的折中得分（越小越接近膝点）
    """

    solution_id: str
    positions: np.ndarray
    objectives: ObjectiveSet
    cable_network: CableNetwork
    crowding_distance: float = 0.0
    normalized: dict[str, float] = field(default_factory=dict)
    weighted_score: float = 0.0


@dataclass
class MultiObjectiveConfig:
    """多目标优化配置。"""

    enabled: bool = False
    algorithm: str = "nsga2"          # nsga2 | mopso
    population_size: int = 40
    max_iterations: int = 60
    archive_size: Optional[int] = 40
    min_spacing_multiple: float = 5.0
    seed: Optional[int] = 42
    # 膝点偏好权重（无需归一化，内部会按总和归一）
    knee_weights: dict = field(default_factory=lambda: {
        "aep": 1.0 / 3.0,
        "lcoe": 1.0 / 3.0,
        "cable": 1.0 / 3.0,
    })
    # 升压站位置 [x, y]；None 表示取各布局机位重心
    substation_position: Optional[list[float]] = None
    # 遗传操作参数（nsga2）
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    mutation_strength: float = 0.1
    # 粒子群参数（mopso）
    inertia_weight: float = 0.7
    cognitive_coeff: float = 1.49
    social_coeff: float = 1.49
    max_velocity: float = 0.2
    turbulence_rate: float = 0.1
    penalty_factor: float = 1e6


@dataclass
class MultiObjectiveResult:
    """多目标优化结果。"""

    solutions: list[ParetoSolution]
    knee_solution: ParetoSolution
    algorithm: str
    knee_weights: dict
    normalization_bounds: dict[str, dict[str, float]]
    history: list[dict]
    substation_fixed: bool
    n_evaluations: int


class LayoutMultiObjectiveEvaluator:
    """布局 → 三目标向量。

    Parameters
    ----------
    aep_evaluate_fn : Callable
        输入 (N,2) 机位，返回净 AEP (MWh/年)，即 ``AEPCalculator.evaluate_layout``
    capital_cost : float
        与布局无关的基准初始投资 (万元)
    om_cost_annual : float
        年运维费用 (万元/年)
    rated_power_MW, n_turbines : float
        单机容量与台数（用于经济模型）
    lcoe_fn : Callable[[float, float, float], float]
        ``EconomicAnalyzer.compute_lcoe``，年发电量参数单位为 GWh
    cable_cost_per_km : float
        集电线路单位造价 (万元/km)，计入初始投资
    substation_position : Optional[np.ndarray]
        固定升压站坐标；None 时按机位重心估算
    """

    def __init__(
        self,
        aep_evaluate_fn: Callable[[np.ndarray], float],
        capital_cost: float,
        om_cost_annual: float,
        lcoe_fn: Callable[[float, float, float], float],
        cable_cost_per_km: float = 80.0,
        substation_position: Optional[np.ndarray] = None,
    ) -> None:
        self.aep_evaluate_fn = aep_evaluate_fn
        self.capital_cost = float(capital_cost)
        self.om_cost_annual = float(om_cost_annual)
        self.lcoe_fn = lcoe_fn
        self.cable_cost_per_km = float(cable_cost_per_km)
        self.substation_position = (
            None if substation_position is None
            else np.asarray(substation_position, dtype=np.float64).reshape(2)
        )
        self.n_evaluations = 0

    def evaluate(self, positions: np.ndarray) -> tuple[ObjectiveSet, CableNetwork]:
        """计算布局的三目标值与集电网络。"""
        self.n_evaluations += 1
        net_aep_mwh = float(self.aep_evaluate_fn(positions))
        network = estimate_cable_network(positions, self.substation_position)

        cable_km = network.total_length / 1000.0
        capital_with_cable = self.capital_cost + cable_km * self.cable_cost_per_km
        lcoe = float(self.lcoe_fn(capital_with_cable, self.om_cost_annual, net_aep_mwh / 1e3))

        return (
            ObjectiveSet(
                net_aep_mwh=net_aep_mwh,
                lcoe=lcoe,
                cable_length_m=network.total_length,
            ),
            network,
        )


# ---------------------------------------------------------------------------
# 非支配排序与拥挤距离
# ---------------------------------------------------------------------------


def non_dominated_sort(
    objectives: np.ndarray,
    feasible: Optional[np.ndarray] = None,
    violations: Optional[np.ndarray] = None,
) -> list[list[int]]:
    """快速非支配排序（最小化方向），支持约束支配。

    约束支配规则：可行解支配不可行解；两个不可行解中违反量小者支配
    违反量大者。

    Returns
    -------
    list[list[int]]
        逐层前沿，每个元素是该层个体在输入中的索引
    """
    n = objectives.shape[0]
    if feasible is None:
        feasible = np.ones(n, dtype=bool)
    if violations is None:
        violations = np.zeros(n, dtype=np.float64)

    def dominates(i: int, j: int) -> bool:
        if feasible[i] and not feasible[j]:
            return True
        if not feasible[i] and feasible[j]:
            return False
        if not feasible[i] and not feasible[j]:
            return violations[i] < violations[j] - 1e-12
        fi, fj = objectives[i], objectives[j]
        le = np.all(fi <= fj + 1e-12)
        lt = np.any(fi < fj - 1e-12)
        return bool(le and lt)

    domination_sets = [[] for _ in range(n)]
    domination_counts = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(p + 1, n):
            if dominates(p, q):
                domination_sets[p].append(q)
                domination_counts[q] += 1
            elif dominates(q, p):
                domination_sets[q].append(p)
                domination_counts[p] += 1
        if domination_counts[p] == 0:
            fronts[0].append(p)

    k = 0
    while fronts[k]:
        next_front = []
        for p in fronts[k]:
            for q in domination_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def crowding_distance(objectives: np.ndarray, front: list[int]) -> np.ndarray:
    """计算一个前沿内各解的拥挤距离。"""
    n = len(front)
    distances = np.zeros(n, dtype=np.float64)
    if n <= 2:
        distances[:] = np.inf
        return distances

    n_obj = objectives.shape[1]
    for m in range(n_obj):
        values = objectives[front, m]
        order = np.argsort(values, kind="stable")
        vmin = values[order[0]]
        vmax = values[order[-1]]
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        span = vmax - vmin
        if span <= 1e-15:
            continue
        for rank in range(1, n - 1):
            idx = order[rank]
            distances[idx] += (
                values[order[rank + 1]] - values[order[rank - 1]]
            ) / span
    return distances


# ---------------------------------------------------------------------------
# 共用的布局约束操作
# ---------------------------------------------------------------------------


class _LayoutOperators:
    """初始化/变异修复等与单目标 GA、PSO 保持一致的约束操作。"""

    def __init__(
        self,
        n_turbines: int,
        rotor_diameters: np.ndarray,
        boundary: SiteBoundary,
        min_spacing_multiple: float,
        rng: np.random.Generator,
        penalty_factor: float = 1e6,
    ) -> None:
        self.n_turbines = n_turbines
        self.rotor_diameters = np.asarray(rotor_diameters, dtype=np.float64)
        self.boundary = boundary
        self.min_spacing = compute_min_spacing_from_diameters(
            self.rotor_diameters, min_spacing_multiple
        )
        self.rng = rng
        self.penalty_factor = penalty_factor
        self.n_dim = n_turbines * 2
        self.x_range = boundary.x_max - boundary.x_min
        self.y_range = boundary.y_max - boundary.y_min

    def generate_valid_layout(self) -> np.ndarray:
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
                return enforce_min_spacing(
                    positions, self.min_spacing, self.boundary, self.rng
                )
            except RuntimeError:
                continue
        raise RuntimeError("无法生成满足约束的初始布局")

    def violation(self, positions: np.ndarray) -> float:
        """返回约束违反量（0 表示完全可行）。"""
        violation = 0.0
        inside = self.boundary.contains_all(positions)
        if not inside.all():
            violation += float(np.sum(~inside)) * self.penalty_factor
        valid, pairs = check_min_spacing(positions, self.min_spacing)
        if not valid:
            for i, j in pairs:
                dist = float(np.linalg.norm(positions[i] - positions[j]))
                violation += (self.min_spacing - dist) * self.penalty_factor
        return violation

    def repair(self, positions_flat: np.ndarray) -> np.ndarray:
        positions = positions_flat.reshape(self.n_turbines, 2)
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


# ---------------------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------------------


class NSGA2Optimizer:
    """带精英保留的多目标遗传算法（NSGA-II）。"""

    def __init__(
        self,
        n_turbines: int,
        rotor_diameters: np.ndarray,
        boundary: SiteBoundary,
        evaluator: LayoutMultiObjectiveEvaluator,
        config: MultiObjectiveConfig,
    ) -> None:
        self.n_turbines = n_turbines
        self.boundary = boundary
        self.evaluator = evaluator
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.ops = _LayoutOperators(
            n_turbines, rotor_diameters, boundary,
            config.min_spacing_multiple, self.rng, config.penalty_factor,
        )

    def _initialize(self, pop_size: int) -> np.ndarray:
        population = np.zeros((pop_size, self.ops.n_dim), dtype=np.float64)
        for i in range(pop_size):
            population[i] = self.ops.generate_valid_layout().flatten()
        return population

    def _evaluate(self, population: np.ndarray):
        n = population.shape[0]
        obj = np.full((n, 3), np.inf, dtype=np.float64)
        feasible = np.ones(n, dtype=bool)
        violations = np.zeros(n, dtype=np.float64)
        networks: list[Optional[CableNetwork]] = [None] * n

        for i in range(n):
            positions = population[i].reshape(self.n_turbines, 2)
            viol = self.ops.violation(positions)
            violations[i] = viol
            if viol > 0:
                feasible[i] = False
                continue
            try:
                obj_set, network = self.evaluator.evaluate(positions)
                obj[i] = obj_set.as_array()
                networks[i] = network
            except Exception:
                feasible[i] = False
                violations[i] += self.config.penalty_factor
        return obj, feasible, violations, networks

    def _tournament(self, population, ranks, crowding, n_select):
        pop_size = population.shape[0]
        selected = np.zeros((n_select, population.shape[1]), dtype=np.float64)
        for i in range(n_select):
            a, b = self.rng.integers(0, pop_size, size=2)
            better = a if (
                ranks[a] < ranks[b]
                or (ranks[a] == ranks[b] and crowding[a] > crowding[b])
            ) else b
            selected[i] = population[better]
        return selected

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        if self.rng.random() > self.config.crossover_rate:
            return p1.copy()
        mask = self.rng.integers(0, 2, size=p1.shape[0], dtype=bool)
        return np.where(mask, p1, p2)

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        mutated = individual.copy()
        for k in range(mutated.shape[0]):
            if self.rng.random() < self.config.mutation_rate:
                span = (
                    self.ops.x_range if k % 2 == 0 else self.ops.y_range
                ) * self.config.mutation_strength
                mutated[k] += self.rng.normal(0.0, span)
        return mutated

    def optimize(self, verbose: bool = True):
        pop_size = self.config.population_size
        max_gen = self.config.max_iterations

        if verbose:
            print("\n=== NSGA-II 多目标优化开始 ===")
            print(f"种群大小: {pop_size}  最大代数: {max_gen}")
            print("目标: 最大化净AEP / 最小化LCOE / 最小化集电线路长度")
            print("=" * 40)

        population = self._initialize(pop_size)
        obj, feasible, violations, networks = self._evaluate(population)

        history = []

        def assign_ranks_crowding(obj_arr, feas_arr, viol_arr):
            fronts = non_dominated_sort(obj_arr, feas_arr, viol_arr)
            ranks = np.zeros(obj_arr.shape[0], dtype=int)
            crowd = np.zeros(obj_arr.shape[0], dtype=np.float64)
            for r, front in enumerate(fronts):
                cd = crowding_distance(obj_arr, front)
                for local, idx in enumerate(front):
                    ranks[idx] = r
                    crowd[idx] = cd[local]
            return fronts, ranks, crowd

        for gen in range(max_gen):
            fronts, ranks, crowd = assign_ranks_crowding(obj, feasible, violations)
            self._record_history(history, fronts, obj, feasible, gen)

            parents = self._tournament(population, ranks, crowd, pop_size)
            offspring = np.zeros_like(population)
            for i in range(0, pop_size, 2):
                p1 = parents[i]
                p2 = parents[(i + 1) % pop_size]
                offspring[i] = self.ops.repair(self._mutate(self._crossover(p1, p2)))
                if i + 1 < pop_size:
                    offspring[i + 1] = self.ops.repair(
                        self._mutate(self._crossover(p2, p1))
                    )

            off_obj, off_feas, off_viol, off_networks = self._evaluate(offspring)

            combined = np.vstack([population, offspring])
            combined_obj = np.vstack([obj, off_obj])
            combined_feas = np.concatenate([feasible, off_feas])
            combined_viol = np.concatenate([violations, off_viol])
            combined_networks = networks + off_networks

            fronts = non_dominated_sort(combined_obj, combined_feas, combined_viol)
            chosen: list[int] = []
            new_crowd = np.zeros(combined.shape[0], dtype=np.float64)
            for front in fronts:
                cd = crowding_distance(combined_obj, front)
                for local, idx in enumerate(front):
                    new_crowd[idx] = cd[local]
                if len(chosen) + len(front) <= pop_size:
                    chosen.extend(front)
                else:
                    need = pop_size - len(chosen)
                    order = np.argsort(-cd, kind="stable")
                    chosen.extend([front[k] for k in order[:need]])
                    break

            idx = np.array(chosen, dtype=int)
            population = combined[idx]
            obj = combined_obj[idx]
            feasible = combined_feas[idx]
            violations = combined_viol[idx]
            networks = [combined_networks[k] for k in idx]

            if verbose and (gen % 5 == 0 or gen == max_gen - 1):
                n0 = len(fronts[0]) if fronts else 0
                print(
                    f"Gen {gen+1:3d} | 非支配解: {n0:3d} | "
                    f"最佳净AEP: {history[-1]['best_aep_mwh']/1e3:7.2f} GWh | "
                    f"最短线缆: {history[-1]['min_cable_m']/1e3:7.2f} km"
                )

        # 最终：在最终种群上取第一前沿（全部可行）
        final_fronts, _, final_crowd = assign_ranks_crowding(obj, feasible, violations)
        first_front = [i for i in final_fronts[0] if feasible[i]]

        # 归档规模限制：保留拥挤距离最大（目标空间最分散）的解
        if self.config.archive_size is not None and len(first_front) > self.config.archive_size:
            front_crowd = np.array([final_crowd[i] for i in first_front])
            keep = np.argsort(-front_crowd, kind="stable")[: self.config.archive_size]
            first_front = [first_front[k] for k in keep]

        records = []
        for i in first_front:
            records.append({
                "positions": population[i].reshape(self.n_turbines, 2).copy(),
                "objectives": combined_to_objectiveset(obj[i]),
                "network": networks[i],
                "crowding": float(final_crowd[i]),
            })

        if verbose:
            print(f"优化完成，Pareto 非支配解数量: {len(records)}")
        return records, history

    @staticmethod
    def _record_history(history, fronts, obj, feasible, gen):
        first = [i for i in fronts[0] if feasible[i]] if fronts else []
        if first:
            aep = float(-obj[first, 0].min())  # 最小化向量中存的是 -AEP
            entry = {
                "iteration": gen + 1,
                "n_non_dominated": len(first),
                "best_aep_mwh": aep,
                "min_lcoe": float(obj[first, 1].min()),
                "min_cable_m": float(obj[first, 2].min()),
            }
        else:
            entry = {
                "iteration": gen + 1,
                "n_non_dominated": 0,
                "best_aep_mwh": 0.0,
                "min_lcoe": float("inf"),
                "min_cable_m": float("inf"),
            }
        history.append(entry)


# ---------------------------------------------------------------------------
# 多目标粒子群 (MOPSO)
# ---------------------------------------------------------------------------


class MOPSOOptimizer:
    """基于 Pareto 归档与拥挤距离领导选择的多目标粒子群。"""

    def __init__(
        self,
        n_turbines: int,
        rotor_diameters: np.ndarray,
        boundary: SiteBoundary,
        evaluator: LayoutMultiObjectiveEvaluator,
        config: MultiObjectiveConfig,
    ) -> None:
        self.n_turbines = n_turbines
        self.boundary = boundary
        self.evaluator = evaluator
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.ops = _LayoutOperators(
            n_turbines, rotor_diameters, boundary,
            config.min_spacing_multiple, self.rng, config.penalty_factor,
        )
        n_dim = self.ops.n_dim
        self.pos_bounds = np.zeros((n_dim, 2), dtype=np.float64)
        for k in range(n_dim):
            if k % 2 == 0:
                self.pos_bounds[k] = [boundary.x_min, boundary.x_max]
            else:
                self.pos_bounds[k] = [boundary.y_min, boundary.y_max]
        self.vel_range = np.zeros(n_dim, dtype=np.float64)
        for k in range(n_dim):
            span = self.ops.x_range if k % 2 == 0 else self.ops.y_range
            self.vel_range[k] = span * config.max_velocity

    def _evaluate(self, positions):
        n = positions.shape[0]
        obj = np.full((n, 3), np.inf, dtype=np.float64)
        feasible = np.ones(n, dtype=bool)
        violations = np.zeros(n, dtype=np.float64)
        networks: list[Optional[CableNetwork]] = [None] * n
        for i in range(n):
            pos = positions[i].reshape(self.n_turbines, 2)
            viol = self.ops.violation(pos)
            violations[i] = viol
            if viol > 0:
                feasible[i] = False
                continue
            try:
                obj_set, network = self.evaluator.evaluate(pos)
                obj[i] = obj_set.as_array()
                networks[i] = network
            except Exception:
                feasible[i] = False
                violations[i] += self.config.penalty_factor
        return obj, feasible, violations, networks

    def _dominates_rows(self, a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.all(a <= b + 1e-12) and np.any(a < b - 1e-12))

    def _update_archive(self, archive, archive_obj, archive_net, positions, obj, networks,
                        feasible, violations, max_size):
        """把可行新解并入归档并做非支配截断。"""
        rec_pos = list(archive)
        rec_obj = list(archive_obj)
        rec_net = list(archive_net)
        for i in range(positions.shape[0]):
            if not feasible[i]:
                continue
            rec_pos.append(positions[i].copy())
            rec_obj.append(obj[i].copy())
            rec_net.append(networks[i])

        if not rec_pos:
            return rec_pos, rec_obj, rec_net, np.zeros(0)

        pos_arr = np.array(rec_pos)
        obj_arr = np.array(rec_obj)
        fronts = non_dominated_sort(obj_arr)
        first = fronts[0]
        crowd = crowding_distance(obj_arr, first)

        if max_size is not None and len(first) > max_size:
            order = np.argsort(-crowd, kind="stable")[:max_size]
            first = [first[k] for k in order]
            crowd = crowd[np.argsort(-crowd, kind="stable")[:max_size]]

        pos_arr = pos_arr[first]
        obj_arr = obj_arr[first]
        net_arr = [rec_net[i] for i in first]
        return list(pos_arr), list(obj_arr), net_arr, crowd

    def optimize(self, verbose: bool = True):
        swarm_size = self.config.population_size
        max_iter = self.config.max_iterations

        if verbose:
            print("\n=== MOPSO 多目标粒子群优化开始 ===")
            print(f"粒子群大小: {swarm_size}  最大迭代: {max_iter}")
            print("目标: 最大化净AEP / 最小化LCOE / 最小化集电线路长度")
            print("=" * 40)

        positions = np.zeros((swarm_size, self.ops.n_dim), dtype=np.float64)
        velocities = np.zeros_like(positions)
        for i in range(swarm_size):
            positions[i] = self.ops.generate_valid_layout().flatten()
            velocities[i] = self.rng.uniform(
                -self.vel_range, self.vel_range, self.ops.n_dim
            )

        obj, feasible, violations, networks = self._evaluate(positions)

        pbest_pos = positions.copy()
        pbest_obj = obj.copy()
        pbest_feas = feasible.copy()

        archive, archive_obj, archive_net, archive_crowd = self._update_archive(
            [], np.zeros((0, 3)), [], positions, obj, networks,
            feasible, violations, self.config.archive_size,
        )

        history = []

        def select_leader():
            # 拥挤距离越大越优先；边界解为 inf，按 1/距离 轮盘赌
            cd = np.where(np.isinf(archive_crowd), 1e6, archive_crowd)
            if cd.sum() <= 0:
                probs = np.ones(len(cd)) / len(cd)
            else:
                probs = cd / cd.sum()
            return int(self.rng.choice(len(archive), p=probs))

        for it in range(max_iter):
            if archive:
                history.append({
                    "iteration": it + 1,
                    "n_non_dominated": len(archive),
                    "best_aep_mwh": float(max(-o[0] for o in archive_obj)),
                    "min_lcoe": float(min(o[1] for o in archive_obj)),
                    "min_cable_m": float(min(o[2] for o in archive_obj)),
                })
            else:
                history.append({
                    "iteration": it + 1, "n_non_dominated": 0,
                    "best_aep_mwh": 0.0,
                    "min_lcoe": float("inf"), "min_cable_m": float("inf"),
                })

            leaders = np.array(
                [archive[select_leader()] if archive else positions[0]
                 for _ in range(swarm_size)]
            )

            r1 = self.rng.random((swarm_size, self.ops.n_dim))
            r2 = self.rng.random((swarm_size, self.ops.n_dim))
            w = self.config.inertia_weight
            c1 = self.config.cognitive_coeff
            c2 = self.config.social_coeff
            velocities = (
                w * velocities
                + c1 * r1 * (pbest_pos - positions)
                + c2 * r2 * (leaders - positions)
            )
            velocities = np.clip(velocities, -self.vel_range, self.vel_range)
            positions = positions + velocities
            positions = np.clip(
                positions, self.pos_bounds[:, 0], self.pos_bounds[:, 1]
            )

            # 湍流：少量维度随机重置，保持探索
            for i in range(swarm_size):
                for k in range(self.ops.n_dim):
                    if self.rng.random() < self.config.turbulence_rate / self.ops.n_dim:
                        positions[i, k] = self.rng.uniform(
                            self.pos_bounds[k, 0], self.pos_bounds[k, 1]
                        )
                positions[i] = self.ops.repair(positions[i])

            obj, feasible, violations, networks = self._evaluate(positions)

            for i in range(swarm_size):
                if feasible[i] and (
                    not pbest_feas[i] or self._dominates_rows(obj[i], pbest_obj[i])
                ):
                    pbest_pos[i] = positions[i].copy()
                    pbest_obj[i] = obj[i].copy()
                    pbest_feas[i] = True
                # 互不支配时保留原个体最优，维持稳定

            archive, archive_obj, archive_net, archive_crowd = self._update_archive(
                archive, archive_obj, archive_net, positions, obj, networks,
                feasible, violations, self.config.archive_size,
            )

            if verbose and (it % 5 == 0 or it == max_iter - 1):
                h = history[-1]
                print(
                    f"Iter {it+1:3d} | 非支配解: {h['n_non_dominated']:3d} | "
                    f"最佳净AEP: {h['best_aep_mwh']/1e3:7.2f} GWh | "
                    f"最短线缆: {h['min_cable_m']/1e3:7.2f} km"
                )

        records = []
        for pos, o, net, cd in zip(archive, archive_obj, archive_net, archive_crowd):
            records.append({
                "positions": pos.reshape(self.n_turbines, 2).copy(),
                "objectives": combined_to_objectiveset(o),
                "network": net,
                "crowding": float(cd),
            })
        if verbose:
            print(f"优化完成，Pareto 非支配解数量: {len(records)}")
        return records, history


# ---------------------------------------------------------------------------
# 结果组装：稳定排序、编号、归一化与膝点
# ---------------------------------------------------------------------------


def combined_to_objectiveset(min_vector: np.ndarray) -> ObjectiveSet:
    return ObjectiveSet(
        net_aep_mwh=float(-min_vector[0]),
        lcoe=float(min_vector[1]),
        cable_length_m=float(min_vector[2]),
    )


def _position_fingerprint(positions: np.ndarray) -> float:
    """机位布局的确定性指纹，用于目标值完全相同时的稳定排序。"""
    rounded = np.round(np.asarray(positions, dtype=np.float64), 6)
    return float(rounded.sum())


def build_pareto_result(
    records: list[dict],
    history: list[dict],
    config: MultiObjectiveConfig,
    n_evaluations: int,
) -> MultiObjectiveResult:
    """对原始非支配记录去重、稳定编号、归一化并选出膝点方案。"""
    # 去重：目标向量几乎相同且机位指纹相同视为同一解，保留先出现者
    unique = []
    seen = set()
    for rec in records:
        o = rec["objectives"]
        key = (
            round(o.net_aep_mwh, 6),
            round(o.lcoe, 9),
            round(o.cable_length_m, 4),
            round(_position_fingerprint(rec["positions"]), 4),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(rec)

    # 确定性排序：净AEP 降序，其次 LCOE 升序、线缆升序、机位指纹升序
    unique.sort(key=lambda r: (
        -r["objectives"].net_aep_mwh,
        r["objectives"].lcoe,
        r["objectives"].cable_length_m,
        _position_fingerprint(r["positions"]),
    ))

    aeps = np.array([r["objectives"].net_aep_mwh for r in unique])
    lcoes = np.array([r["objectives"].lcoe for r in unique])
    cables = np.array([r["objectives"].cable_length_m for r in unique])

    def norm_high(v, arr):
        span = arr.max() - arr.min()
        return (v - arr.min()) / span if span > 1e-15 else 1.0

    def norm_low(v, arr):
        span = arr.max() - arr.min()
        return (arr.max() - v) / span if span > 1e-15 else 1.0

    raw_weights = config.knee_weights
    w_sum = sum(raw_weights.get(k, 0.0) for k in ("aep", "lcoe", "cable"))
    if w_sum <= 0:
        raise ValueError("膝点偏好权重之和必须为正")
    w = {
        "aep": raw_weights.get("aep", 0.0) / w_sum,
        "lcoe": raw_weights.get("lcoe", 0.0) / w_sum,
        "cable": raw_weights.get("cable", 0.0) / w_sum,
    }

    solutions: list[ParetoSolution] = []
    for i, rec in enumerate(unique):
        o = rec["objectives"]
        normalized = {
            "aep": float(norm_high(o.net_aep_mwh, aeps)),
            "lcoe": float(norm_low(o.lcoe, lcoes)),
            "cable": float(norm_low(o.cable_length_m, cables)),
        }
        # 归一化后 1 为最好；折中的"差距"为 1-norm，加权求和取最小
        score = (
            w["aep"] * (1.0 - normalized["aep"])
            + w["lcoe"] * (1.0 - normalized["lcoe"])
            + w["cable"] * (1.0 - normalized["cable"])
        )
        solutions.append(ParetoSolution(
            solution_id=f"P{i:02d}",
            positions=rec["positions"],
            objectives=o,
            cable_network=rec["network"],
            crowding_distance=rec["crowding"],
            normalized=normalized,
            weighted_score=float(score),
        ))

    # 膝点：加权归一化距离最小；并列时取编号最靠前（确定性）
    knee_idx = int(np.argmin([s.weighted_score for s in solutions]))
    knee = solutions[knee_idx]

    bounds = {
        "net_aep_mwh": {"min": float(aeps.min()), "max": float(aeps.max())},
        "net_aep_gwh": {"min": float(aeps.min() / 1e3), "max": float(aeps.max() / 1e3)},
        "lcoe": {"min": float(lcoes.min()), "max": float(lcoes.max())},
        "cable_length_m": {"min": float(cables.min()), "max": float(cables.max())},
        "cable_length_km": {"min": float(cables.min() / 1e3), "max": float(cables.max() / 1e3)},
    }

    return MultiObjectiveResult(
        solutions=solutions,
        knee_solution=knee,
        algorithm=config.algorithm,
        knee_weights=w,
        normalization_bounds=bounds,
        history=history,
        substation_fixed=config.substation_position is not None,
        n_evaluations=n_evaluations,
    )


def run_multi_objective(
    n_turbines: int,
    rotor_diameters: np.ndarray,
    boundary: SiteBoundary,
    evaluator: LayoutMultiObjectiveEvaluator,
    config: MultiObjectiveConfig,
    verbose: bool = True,
) -> MultiObjectiveResult:
    """按配置运行多目标优化并返回组装好的 Pareto 结果。"""
    algo = config.algorithm.lower()
    if algo == "nsga2":
        optimizer = NSGA2Optimizer(n_turbines, rotor_diameters, boundary, evaluator, config)
    elif algo == "mopso":
        optimizer = MOPSOOptimizer(n_turbines, rotor_diameters, boundary, evaluator, config)
    else:
        raise ValueError(f"未知的多目标算法: {algo}（支持 nsga2 / mopso）")

    records, history = optimizer.optimize(verbose=verbose)
    if not records:
        raise RuntimeError("多目标优化未能得到任何可行非支配解，请检查场地与间距约束")
    return build_pareto_result(records, history, config, evaluator.n_evaluations)
