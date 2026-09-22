"""多目标优化的目标评估器。

三个目标同时用于多目标模式：

1. ``net_aep_mwh`` —— 净年发电量，越大越好（MWh/年）；
2. ``lcoe`` —— 度电成本，越小越好（元/kWh），其初始投资在原有
   经济性模型基础上额外计入随布局变化的集电系统投资；
3. ``collection_length_m`` —— 机位到升压站连通网络(MST)的集电线路
   总长度，越小越好 (m)。

所有目标均保留原始物理量纲，优化器内部通过方向标记与外部归一化处理量纲差异。
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..constraints.boundary import SiteBoundary
from ..economy.costs import EconomicAnalyzer
from ..farm.aep import AEPCalculator
from ..farm.collection import (
    CollectionCostModel,
    CollectionNetwork,
    collection_capital_cost,
    estimate_collection_network,
)


# 目标键的固定顺序，同时决定 Pareto 结果文件与图形中的列顺序。
OBJECTIVE_KEYS: tuple[str, ...] = (
    "net_aep_mwh",
    "lcoe_yuan_per_kwh",
    "collection_length_m",
)

# 各目标的优化方向：+1 表示越大越好，-1 表示越小越好。
OBJECTIVE_DIRECTIONS: dict[str, int] = {
    "net_aep_mwh": +1,
    "lcoe_yuan_per_kwh": -1,
    "collection_length_m": -1,
}

OBJECTIVE_LABELS: dict[str, str] = {
    "net_aep_mwh": "净AEP (MWh/年)",
    "lcoe_yuan_per_kwh": "度电成本 LCOE (元/kWh)",
    "collection_length_m": "集电线路长度 (m)",
}


@dataclass
class LayoutObjectives:
    """单个布局的目标值与可回溯明细。"""

    net_aep_mwh: float
    lcoe_yuan_per_kwh: float
    collection_length_m: float
    collection_cost_wanyuan: float
    total_capital_cost_wanyuan: float
    network: Optional[CollectionNetwork] = None

    def as_dict(self) -> dict[str, float]:
        """按固定顺序返回三个目标的原始量纲值。"""
        return {
            "net_aep_mwh": float(self.net_aep_mwh),
            "lcoe_yuan_per_kwh": float(self.lcoe_yuan_per_kwh),
            "collection_length_m": float(self.collection_length_m),
        }


class MultiObjectiveEvaluator:
    """布局的三目标评估器。

    Parameters
    ----------
    aep_calculator : AEPCalculator
        净AEP计算器
    economic_analyzer : EconomicAnalyzer
        经济性分析器
    n_turbines : int
        风机台数
    rated_power_per_turbine_MW : float
        单台额定功率 (MW)
    boundary : SiteBoundary
        场地边界（用于缺省升压站选址）
    substation_xy : Optional[np.ndarray]
        升压站坐标；缺省取场地质心（顶点均值）
    collection_cost_model : Optional[CollectionCostModel]
        集电系统造价模型
    """

    def __init__(
        self,
        aep_calculator: AEPCalculator,
        economic_analyzer: EconomicAnalyzer,
        n_turbines: int,
        rated_power_per_turbine_MW: float,
        boundary: SiteBoundary,
        substation_xy: Optional[np.ndarray] = None,
        collection_cost_model: Optional[CollectionCostModel] = None,
    ) -> None:
        self.aep_calculator = aep_calculator
        self.economic_analyzer = economic_analyzer
        self.n_turbines = n_turbines
        self.rated_power_per_turbine_MW = rated_power_per_turbine_MW
        self.boundary = boundary
        self.collection_cost_model = (
            collection_cost_model
            if collection_cost_model is not None
            else CollectionCostModel()
        )

        if substation_xy is None:
            # 顶点均值是凸多边形质心的稳定近似，且对任意多边形都有定义。
            substation_xy = np.mean(boundary.vertices, axis=0)
        self.substation_xy = np.asarray(substation_xy, dtype=np.float64).reshape(2)

    def evaluate(
        self,
        positions: np.ndarray,
        retain_network: bool = True,
    ) -> LayoutObjectives:
        """计算单个布局的全部目标。

        Parameters
        ----------
        positions : np.ndarray
            风机位置 (N, 2)
        retain_network : bool
            是否在结果中保留集电网络明细（结果输出需要，排序时可关闭以省内存）
        """
        positions = np.asarray(positions, dtype=np.float64)

        net_aep_mwh = float(self.aep_calculator.evaluate_layout(positions))

        network = estimate_collection_network(positions, self.substation_xy)

        base_capital, _ = self.economic_analyzer.compute_capital_cost(
            self.n_turbines, self.rated_power_per_turbine_MW
        )
        annual_om = self.economic_analyzer.compute_annual_om_cost(
            self.n_turbines, self.rated_power_per_turbine_MW
        )
        collection_cost, _ = collection_capital_cost(
            network, self.n_turbines, self.collection_cost_model
        )
        total_capital = base_capital + collection_cost

        lcoe = self.economic_analyzer.compute_lcoe(
            total_capital,
            annual_om,
            net_aep_mwh / 1e3,
        )

        return LayoutObjectives(
            net_aep_mwh=net_aep_mwh,
            lcoe_yuan_per_kwh=float(lcoe),
            collection_length_m=float(network.total_length_m),
            collection_cost_wanyuan=float(collection_cost),
            total_capital_cost_wanyuan=float(total_capital),
            network=network if retain_network else None,
        )


def objectives_to_minimize_vector(values: LayoutObjectives) -> np.ndarray:
    """把目标值转换为统一的"越小越好"向量，用于非支配排序。"""
    d = values.as_dict()
    return np.array(
        [OBJECTIVE_DIRECTIONS[k] * d[k] for k in OBJECTIVE_KEYS],
        dtype=np.float64,
    )
