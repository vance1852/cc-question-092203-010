"""命令行接口。

提供完整的风电场机位布局评估和优化流程。
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Optional

import numpy as np

from .config import WindFarmConfig, create_sample_config
from .core.turbine import Turbine
from .core.wind_resource import WindResource
from .core.wake import WakeModel
from .constraints.boundary import SiteBoundary
from .farm.aep import AEPCalculator, FarmResult
from .farm.collection import CollectionCostModel
from .optimization.baseline import generate_grid_layout
from .optimization.ga import GeneticAlgorithm, GAConfig
from .optimization.pso import ParticleSwarmOptimizer, PSOConfig
from .optimization.multiobjective import (
    MultiObjectiveOptimizer,
    MultiObjectiveResult,
)
from .optimization.objectives import (
    OBJECTIVE_KEYS,
    OBJECTIVE_LABELS,
    MultiObjectiveEvaluator,
)
from .economy.costs import (
    EconomicAnalyzer,
    EconomicResult,
    get_default_turbine_cost,
    get_default_farm_cost,
)
from .visualization.plotting import (
    plot_farm_layout,
    plot_wind_rose,
    plot_convergence,
    plot_aep_vs_turbines,
    plot_turbine_loss_bar,
    plot_comparison,
    plot_wake_heatmap,
)
from .visualization.multiobj_plot import (
    plot_pareto_pairwise,
    plot_pareto_parallel,
    plot_pareto_convergence,
    plot_collection_network,
)


class WindFarmOptimizerCLI:
    """风电场优化命令行接口主类。"""

    def __init__(self, config: WindFarmConfig) -> None:
        self.config = config
        self._setup_output_dir()

        self.turbines = config.create_turbines()
        self.boundary = config.create_boundary()
        self.wind_resource = config.create_wind_resource()
        self.wake_model = config.create_wake_model()

        self.rotor_diameters = np.array([t.rotor_diameter for t in self.turbines])
        self.rated_powers = np.array([t.rated_power for t in self.turbines])
        self.thrust_coefficients = np.array([t.thrust_coefficient for t in self.turbines])

        self.aep_calc = AEPCalculator(
            turbines=self.turbines,
            wind_resource=self.wind_resource,
            wake_model=self.wake_model,
            wake_superposition=config.superposition_method,
        )

        self.baseline_positions: Optional[np.ndarray] = None
        self.baseline_result: Optional[FarmResult] = None
        self.optimized_positions: Optional[np.ndarray] = None
        self.optimized_result: Optional[FarmResult] = None
        self.optimize_result = None
        self.economic_result: Optional[EconomicResult] = None
        self.sweep_results: Optional[dict] = None
        self.multi_objective_result: Optional[MultiObjectiveResult] = None

    def _setup_output_dir(self) -> None:
        """创建输出目录。"""
        output_dir = self.config.visualization.save_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(f"输出目录: {os.path.abspath(output_dir)}")

    def _print_header(self, title: str) -> None:
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)

    def _print_result_summary(self, result: FarmResult, label: str = "") -> None:
        """打印计算结果摘要。"""
        print(f"\n--- {label} 结果 ---")
        print(f"  装机容量:    {result.total_installed_capacity:.2f} MW")
        print(f"  理论AEP:     {result.gross_aep/1e3:.2f} GWh/年")
        print(f"  净AEP:       {result.net_aep/1e3:.2f} GWh/年")
        print(f"  尾流损失:    {result.total_wake_loss/1e3:.2f} GWh/年 ({result.wake_loss_pct:.2f}%)")
        print(f"  容量系数:    {result.capacity_factor:.2f}%")
        print(f"  风机台数:    {len(result.turbine_results)}")

        max_loss_turb = max(result.turbine_results, key=lambda x: x.wake_loss_pct)
        print(f"  最大损失风机: #{max_loss_turb.turbine_idx} ({max_loss_turb.wake_loss_pct:.2f}%)")
        if max_loss_turb.dominant_wake_source is not None:
            print(f"    主要影响源: #{max_loss_turb.dominant_wake_source}")

    def run_baseline(self) -> None:
        """运行基线（规则网格布局）评估。"""
        self._print_header("步骤 1/6: 生成并评估基线网格布局")

        rng = np.random.default_rng(self.config.optimization.seed)
        self.baseline_positions = generate_grid_layout(
            boundary=self.boundary,
            n_turbines=self.config.n_turbines,
            rotor_diameters=self.rotor_diameters,
            min_multiple=self.config.optimization.min_spacing_multiple,
            rng=rng,
        )

        print(f"已生成 {self.config.n_turbines} 台风机的网格布局")

        self.baseline_result = self.aep_calc.compute_farm_aep(self.baseline_positions)
        self._print_result_summary(self.baseline_result, "基线布局")

    def run_optimization(self) -> None:
        """运行机位优化。"""
        self._print_header("步骤 2/6: 执行机位布局优化")

        fit_fn = self.aep_calc.evaluate_layout

        algo = self.config.optimization.algorithm.lower()

        if algo == "ga":
            ga_config = GAConfig(
                population_size=self.config.optimization.population_size,
                max_generations=self.config.optimization.max_iterations,
                min_spacing_multiple=self.config.optimization.min_spacing_multiple,
                seed=self.config.optimization.seed,
            )
            optimizer = GeneticAlgorithm(
                n_turbines=self.config.n_turbines,
                rotor_diameters=self.rotor_diameters,
                boundary=self.boundary,
                fitness_fn=fit_fn,
                config=ga_config,
            )
        elif algo == "pso":
            pso_config = PSOConfig(
                swarm_size=self.config.optimization.population_size,
                max_iterations=self.config.optimization.max_iterations,
                min_spacing_multiple=self.config.optimization.min_spacing_multiple,
                seed=self.config.optimization.seed,
            )
            optimizer = ParticleSwarmOptimizer(
                n_turbines=self.config.n_turbines,
                rotor_diameters=self.rotor_diameters,
                boundary=self.boundary,
                fitness_fn=fit_fn,
                config=pso_config,
            )
        else:
            raise ValueError(f"未知的优化算法: {algo}")

        print(f"使用优化算法: {algo.upper()}")
        self.optimize_result = optimizer.optimize(verbose=True)

        self.optimized_positions = self.optimize_result.best_positions
        self.optimized_result = self.aep_calc.compute_farm_aep(self.optimized_positions)

        print("\n--- 优化后结果 ---")
        self._print_result_summary(self.optimized_result, "优化后布局")

        if self.baseline_result is not None:
            improvement = (
                (self.optimized_result.net_aep - self.baseline_result.net_aep)
                / self.baseline_result.net_aep
                * 100
            )
            loss_reduction = (
                (self.baseline_result.wake_loss_pct - self.optimized_result.wake_loss_pct)
                / self.baseline_result.wake_loss_pct
                * 100
            )
            print(f"\n--- 优化提升 ---")
            print(f"  发电量提升:   {improvement:+.2f}%")
            print(f"  尾流损失减少: {loss_reduction:+.2f}%")
            print(f"  额外发电量:   {(self.optimized_result.net_aep - self.baseline_result.net_aep)/1e3:+.2f} GWh/年")

    def run_multi_objective_optimization(self) -> None:
        """运行 NSGA-II 多目标机位优化。"""
        self._print_header("步骤 2/6: 执行多目标布局优化 (NSGA-II)")

        mo_cfg = self.config.multi_objective

        turbine_cost = get_default_turbine_cost(self.config.turbine_model)
        farm_cost = get_default_farm_cost()
        farm_cost.discount_rate = self.config.economic.discount_rate
        analyzer = EconomicAnalyzer(
            turbine_cost=turbine_cost,
            farm_cost=farm_cost,
            electricity_price=self.config.economic.electricity_price,
        )

        substation_xy = (
            np.asarray(mo_cfg.substation_xy, dtype=np.float64)
            if mo_cfg.substation_xy is not None
            else None
        )

        collection_cost_model = CollectionCostModel(
            cable_cost_per_km_wanyuan=mo_cfg.cable_cost_per_km_wanyuan
        )

        evaluator = MultiObjectiveEvaluator(
            aep_calculator=self.aep_calc,
            economic_analyzer=analyzer,
            n_turbines=self.config.n_turbines,
            rated_power_per_turbine_MW=self.turbines[0].rated_power / 1e3,
            boundary=self.boundary,
            substation_xy=substation_xy,
            collection_cost_model=collection_cost_model,
        )

        from .optimization.multiobjective import MultiObjectiveConfig as MOConfig

        mo_alg_config = MOConfig(
            population_size=mo_cfg.population_size,
            max_generations=mo_cfg.max_generations,
            archive_size=mo_cfg.archive_size,
            min_spacing_multiple=self.config.optimization.min_spacing_multiple,
            seed=self.config.optimization.seed,
            preference_weights=mo_cfg.preference_weights,
        )

        optimizer = MultiObjectiveOptimizer(
            n_turbines=self.config.n_turbines,
            rotor_diameters=self.rotor_diameters,
            boundary=self.boundary,
            evaluator=evaluator,
            config=mo_alg_config,
            algorithm="nsga2",
        )

        self.multi_objective_result = optimizer.optimize(verbose=True)

        # 膝点方案同时作为"优化后布局"，便于沿用既有经济性/对比输出。
        knee = self.multi_objective_result.knee_solution
        self.optimized_positions = knee.positions
        self.optimized_result = self.aep_calc.compute_farm_aep(knee.positions)
        # 单目标风格的收敛对象不再生成；optimize_result 保持 None，
        # 避免误画单目标收敛曲线。

        print("\n--- Pareto 解集摘要（按净AEP降序的确定性编号） ---")
        print(f"  共 {len(self.multi_objective_result.solutions)} 个非支配方案")
        print(f"  {'编号':>4} | {'净AEP(GWh)':>11} | {'LCOE(元/kWh)':>13} | "
              f"{'集电线路(km)':>13}")
        for s in self.multi_objective_result.solutions:
            d = s.objectives.as_dict()
            mark = " *膝点" if s.solution_id == knee.solution_id else ""
            print(
                f"  {s.solution_id:>4} | {d['net_aep_mwh']/1e3:11.2f} | "
                f"{d['lcoe_yuan_per_kwh']:13.4f} | "
                f"{d['collection_length_m']/1e3:13.2f}{mark}"
            )

        w = self.multi_objective_result.preference_weights
        print(
            "\n  膝点归一化偏好权重: "
            f"AEP={w['net_aep_mwh']:.3f}, "
            f"LCOE={w['lcoe_yuan_per_kwh']:.3f}, "
            f"线路={w['collection_length_m']:.3f}（内部已归一化）"
        )
        print("\n--- 膝点方案全场评估 ---")
        self._print_result_summary(self.optimized_result, "膝点布局")

        if self.baseline_result is not None:
            improvement = (
                (self.optimized_result.net_aep - self.baseline_result.net_aep)
                / self.baseline_result.net_aep * 100
            )
            print(f"\n--- 膝点方案相对基线 ---")
            print(f"  发电量提升: {improvement:+.2f}%")
            print(
                f"  额外发电量: "
                f"{(self.optimized_result.net_aep - self.baseline_result.net_aep)/1e3:+.2f} GWh/年"
            )

    def run_economic_analysis(self) -> None:
        """运行经济性分析。"""
        if not self.config.economic.enable_analysis:
            return

        self._print_header("步骤 3/6: 经济性分析")

        if self.optimized_result is None:
            print("警告: 未进行优化，使用基线布局进行经济性分析")
            result = self.baseline_result
        else:
            result = self.optimized_result

        turbine_cost = get_default_turbine_cost(self.config.turbine_model)
        farm_cost = get_default_farm_cost()
        farm_cost.discount_rate = self.config.economic.discount_rate

        analyzer = EconomicAnalyzer(
            turbine_cost=turbine_cost,
            farm_cost=farm_cost,
            electricity_price=self.config.economic.electricity_price,
        )

        rated_power_MW = self.turbines[0].rated_power / 1e3
        self.economic_result = analyzer.analyze(
            n_turbines=self.config.n_turbines,
            rated_power_per_turbine_MW=rated_power_MW,
            net_aep_GWh=result.net_aep / 1e3,
        )

        # 多目标模式：用膝点方案的集电网络投资修正总投资与 LCOE，
        # 使评审打印与 Pareto 目标值完全一致。
        if self.multi_objective_result is not None:
            knee_obj = self.multi_objective_result.knee_solution.objectives
            self.economic_result.total_capital_cost = (
                knee_obj.total_capital_cost_wanyuan
            )
            self.economic_result.lcoe = knee_obj.lcoe_yuan_per_kwh
            collection_cost = knee_obj.collection_cost_wanyuan
            self.economic_result.cost_breakdown["集电系统"] = float(collection_cost)

        print(f"\n--- 经济性分析结果（基于优化后布局） ---")
        print(f"  上网电价:      {self.config.economic.electricity_price:.2f} 元/kWh")
        print(f"  折现率:        {self.config.economic.discount_rate*100:.1f}%")
        print(f"  初始投资:      {self.economic_result.total_capital_cost/1e4:.2f} 亿元")
        print(f"  年运维费用:    {self.economic_result.total_om_cost_annual:.1f} 万元/年")
        print(f"  年发电收益:    {self.economic_result.annual_revenue:.1f} 万元/年")
        print(f"  度电成本:      {self.economic_result.lcoe:.3f} 元/kWh")

        if self.economic_result.npv is not None:
            print(f"  净现值(NPV):   {self.economic_result.npv/1e4:+.2f} 亿元")
        if self.economic_result.irr is not None:
            print(f"  内部收益率:    {self.economic_result.irr:.2f}%")
        if self.economic_result.payback_period is not None:
            print(f"  投资回收期:    {self.economic_result.payback_period:.1f} 年")

        print(f"\n  成本构成:")
        for item, cost in self.economic_result.cost_breakdown.items():
            pct = cost / self.economic_result.total_capital_cost * 100
            print(f"    {item}: {cost/1e4:.2f} 亿元 ({pct:.1f}%)")

    def run_turbine_sweep(self, min_turbines: int = 5, max_turbines: int = 25, step: int = 2) -> None:
        """运行风机台数扫描分析。"""
        self._print_header("步骤 4/6: 风机台数扫描分析")

        print(f"扫描范围: {min_turbines} ~ {max_turbines} 台，步长 {step}")
        print("此分析将为不同台数快速优化布局并评估经济性")

        sweep_data = {
            "n_turbines": [],
            "aep": [],
            "lcoe": [],
        }

        rng = np.random.default_rng(self.config.optimization.seed)
        original_n = self.config.n_turbines

        turbine_cost = get_default_turbine_cost(self.config.turbine_model)
        farm_cost = get_default_farm_cost()
        analyzer = EconomicAnalyzer(
            turbine_cost=turbine_cost,
            farm_cost=farm_cost,
            electricity_price=self.config.economic.electricity_price,
        )

        for n in range(min_turbines, max_turbines + 1, step):
            print(f"\n  分析 {n} 台风机...")
            self.config.n_turbines = n
            self.turbines = [self.turbines[0] for _ in range(n)]
            self.rotor_diameters = np.array([t.rotor_diameter for t in self.turbines])
            self.rated_powers = np.array([t.rated_power for t in self.turbines])

            self.aep_calc = AEPCalculator(
                turbines=self.turbines,
                wind_resource=self.wind_resource,
                wake_model=self.wake_model,
                wake_superposition=self.config.superposition_method,
            )

            try:
                positions = generate_grid_layout(
                    boundary=self.boundary,
                    n_turbines=n,
                    rotor_diameters=self.rotor_diameters,
                    min_multiple=self.config.optimization.min_spacing_multiple,
                    rng=rng,
                )

                result = self.aep_calc.compute_farm_aep(positions)

                rated_power_MW = self.turbines[0].rated_power / 1e3
                econ_result = analyzer.analyze(
                    n_turbines=n,
                    rated_power_per_turbine_MW=rated_power_MW,
                    net_aep_GWh=result.net_aep / 1e3,
                )

                sweep_data["n_turbines"].append(n)
                sweep_data["aep"].append(result.net_aep)
                sweep_data["lcoe"].append(econ_result.lcoe)

                print(f"    净AEP: {result.net_aep/1e3:.1f} GWh, LCOE: {econ_result.lcoe:.3f} 元/kWh")
            except Exception as e:
                print(f"    跳过: {e}")

        self.sweep_results = sweep_data
        self.config.n_turbines = original_n

    def run_visualization(self) -> None:
        """生成所有可视化图表。"""
        self._print_header("步骤 5/6: 生成可视化图表")

        save_dir = self.config.visualization.save_dir
        save = self.config.visualization.save_plots
        show = self.config.visualization.show_plots

        if save:
            print("图表将保存到:", os.path.abspath(save_dir))

        plot_wind_rose(
            wind_resource=self.wind_resource,
            title="项目场址风玫瑰图",
            save_path=os.path.join(save_dir, "wind_rose.png") if save else None,
            show=show,
        )

        if self.baseline_positions is not None and self.baseline_result is not None:
            baseline_losses = np.array([tr.wake_loss_pct for tr in self.baseline_result.turbine_results])
            plot_farm_layout(
                positions=self.baseline_positions,
                boundary=self.boundary,
                rotor_diameters=self.rotor_diameters,
                turbine_losses=baseline_losses,
                turbine_names=[f"#{i}" for i in range(len(self.baseline_positions))],
                title="基线网格布局 - 尾流损失分布",
                save_path=os.path.join(save_dir, "baseline_layout.png") if save else None,
                show=show,
            )

            plot_turbine_loss_bar(
                farm_result=self.baseline_result,
                title="基线布局 - 各风机尾流损失",
                save_path=os.path.join(save_dir, "baseline_losses.png") if save else None,
                show=show,
            )

        if self.optimized_positions is not None and self.optimized_result is not None:
            opt_losses = np.array([tr.wake_loss_pct for tr in self.optimized_result.turbine_results])
            plot_farm_layout(
                positions=self.optimized_positions,
                boundary=self.boundary,
                rotor_diameters=self.rotor_diameters,
                turbine_losses=opt_losses,
                turbine_names=[f"#{i}" for i in range(len(self.optimized_positions))],
                title="优化后布局 - 尾流损失分布",
                save_path=os.path.join(save_dir, "optimized_layout.png") if save else None,
                show=show,
            )

            plot_turbine_loss_bar(
                farm_result=self.optimized_result,
                title="优化后布局 - 各风机尾流损失",
                save_path=os.path.join(save_dir, "optimized_losses.png") if save else None,
                show=show,
            )

        if self.optimize_result is not None and self.baseline_result is not None:
            plot_convergence(
                optimize_result=self.optimize_result,
                baseline_aep=self.baseline_result.net_aep,
                title="优化收敛曲线",
                save_path=os.path.join(save_dir, "convergence.png") if save else None,
                show=show,
            )

        if self.baseline_result is not None and self.optimized_result is not None:
            plot_comparison(
                baseline_result=self.baseline_result,
                optimized_result=self.optimized_result,
                title="优化前后关键指标对比",
                save_path=os.path.join(save_dir, "comparison.png") if save else None,
                show=show,
            )

        if self.sweep_results is not None:
            plot_aep_vs_turbines(
                n_turbines_list=self.sweep_results["n_turbines"],
                aep_list=self.sweep_results["aep"],
                lcoe_list=self.sweep_results["lcoe"],
                title="风机台数优化分析",
                save_path=os.path.join(save_dir, "aep_vs_turbines.png") if save else None,
                show=show,
            )

        if self.config.visualization.plot_wake_heatmap and self.optimized_positions is not None:
            dominant_dir = self.wind_resource.directions[np.argmax(self.wind_resource.frequencies)]
            plot_wake_heatmap(
                positions=self.optimized_positions,
                boundary=self.boundary,
                wake_model=self.wake_model,
                wind_direction=dominant_dir,
                rotor_diameters=self.rotor_diameters,
                thrust_coefficients=self.thrust_coefficients,
                title=f"主风向({dominant_dir:.0f}°)尾流速度亏损分布",
                save_path=os.path.join(save_dir, "wake_heatmap.png") if save else None,
                show=show,
            )

        if self.multi_objective_result is not None:
            mo = self.multi_objective_result
            plot_pareto_pairwise(
                result=mo,
                title="Pareto 非支配解集 - 净AEP / LCOE / 集电线路长度",
                save_path=os.path.join(save_dir, "pareto_pairwise.png") if save else None,
                show=show,
            )
            plot_pareto_parallel(
                result=mo,
                save_path=os.path.join(save_dir, "pareto_parallel.png") if save else None,
                show=show,
            )
            plot_pareto_convergence(
                result=mo,
                save_path=os.path.join(save_dir, "pareto_convergence.png") if save else None,
                show=show,
            )
            plot_collection_network(
                solution=mo.knee_solution,
                boundary=self.boundary,
                rotor_diameters=self.rotor_diameters,
                title=f"膝点方案 {mo.knee_solution.solution_id} 集电网络布局",
                save_path=os.path.join(save_dir, "knee_collection_network.png") if save else None,
                show=show,
            )

    def save_results(self) -> None:
        """保存所有结果到JSON文件。"""
        self._print_header("步骤 6/6: 保存结果数据")

        output_dir = self.config.visualization.save_dir

        results = {
            "config": {
                "n_turbines": self.config.n_turbines,
                "turbine_model": self.config.turbine_model,
                "wake_model": self.config.wake_model,
                "min_spacing_multiple": self.config.optimization.min_spacing_multiple,
            },
            "site": {
                "area_km2": float(self.boundary.area / 1e6),
                "mean_wind_speed": float(self.wind_resource.overall_mean_speed),
            },
        }

        if self.baseline_result is not None:
            results["baseline"] = {
                "positions": self.baseline_positions.tolist() if self.baseline_positions is not None else None,
                "gross_aep_gwh": float(self.baseline_result.gross_aep / 1e3),
                "net_aep_gwh": float(self.baseline_result.net_aep / 1e3),
                "wake_loss_pct": float(self.baseline_result.wake_loss_pct),
                "capacity_factor": float(self.baseline_result.capacity_factor),
                "turbine_losses": [
                    {
                        "idx": tr.turbine_idx,
                        "wake_loss_pct": float(tr.wake_loss_pct),
                        "dominant_source": tr.dominant_wake_source,
                    }
                    for tr in self.baseline_result.turbine_results
                ],
            }

        if self.optimized_result is not None:
            results["optimized"] = {
                "positions": self.optimized_positions.tolist() if self.optimized_positions is not None else None,
                "gross_aep_gwh": float(self.optimized_result.gross_aep / 1e3),
                "net_aep_gwh": float(self.optimized_result.net_aep / 1e3),
                "wake_loss_pct": float(self.optimized_result.wake_loss_pct),
                "capacity_factor": float(self.optimized_result.capacity_factor),
                "turbine_losses": [
                    {
                        "idx": tr.turbine_idx,
                        "wake_loss_pct": float(tr.wake_loss_pct),
                        "dominant_source": tr.dominant_wake_source,
                    }
                    for tr in self.optimized_result.turbine_results
                ],
            }

        if self.economic_result is not None:
            results["economic"] = {
                "total_capital_cost_yiyuan": float(self.economic_result.total_capital_cost / 1e4),
                "annual_revenue_wanyuan": float(self.economic_result.annual_revenue),
                "lcoe_yuan_per_kwh": float(self.economic_result.lcoe),
                "npv_yiyuan": float(self.economic_result.npv / 1e4) if self.economic_result.npv is not None else None,
                "irr_pct": float(self.economic_result.irr) if self.economic_result.irr is not None else None,
                "payback_years": float(self.economic_result.payback_period) if self.economic_result.payback_period is not None else None,
            }

        if self.baseline_result is not None and self.optimized_result is not None:
            results["improvement"] = {
                "aep_improvement_pct": float(
                    (self.optimized_result.net_aep - self.baseline_result.net_aep)
                    / self.baseline_result.net_aep * 100
                ),
                "additional_aep_gwh": float(
                    (self.optimized_result.net_aep - self.baseline_result.net_aep) / 1e3
                ),
                "loss_reduction_pct": float(
                    (self.baseline_result.wake_loss_pct - self.optimized_result.wake_loss_pct)
                    / self.baseline_result.wake_loss_pct * 100
                ),
            }

        if self.sweep_results is not None:
            results["turbine_sweep"] = {
                "n_turbines": self.sweep_results["n_turbines"],
                "aep_mwh": self.sweep_results["aep"],
                "lcoe_yuan_per_kwh": self.sweep_results["lcoe"],
            }

        if self.multi_objective_result is not None:
            results["multi_objective"] = self._build_multi_objective_summary()

        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if self.multi_objective_result is not None:
            self._save_pareto_files(output_dir)

        config_path = os.path.join(output_dir, "config.json")
        self.config.to_json(config_path)

        print(f"结果已保存到: {os.path.abspath(results_path)}")
        if self.multi_objective_result is not None:
            print(f"Pareto 方案已保存到: {os.path.abspath(os.path.join(output_dir, 'pareto_front.json'))}")
            print(f"Pareto 目标表已保存到: {os.path.abspath(os.path.join(output_dir, 'pareto_front.csv'))}")
        print(f"配置已保存到: {os.path.abspath(config_path)}")

    def _build_multi_objective_summary(self) -> dict:
        """构建 results.json 中的多目标摘要（完整方案另存 pareto_front.json）。"""
        mo = self.multi_objective_result

        extremes: dict[str, dict] = {}
        solutions = mo.solutions
        for m, key in enumerate(OBJECTIVE_KEYS):
            vals = [s.objective_vector()[m] for s in solutions]
            d = mo.objective_directions[key]
            if d > 0:
                best_sol = max(solutions, key=lambda s, m=m: s.objective_vector()[m])
                best_value, worst_value = max(vals), min(vals)
            else:
                best_sol = min(solutions, key=lambda s, m=m: s.objective_vector()[m])
                best_value, worst_value = min(vals), max(vals)
            extremes[key] = {
                "label": OBJECTIVE_LABELS[key],
                "direction": "maximize" if d > 0 else "minimize",
                "best_solution_id": best_sol.solution_id,
                "best_value": float(best_value),
                "worst_value": float(worst_value),
            }

        return {
            "algorithm": mo.algorithm,
            "seed": mo.seed,
            "objective_keys": list(mo.objective_keys),
            "objective_labels": mo.objective_labels,
            "objective_directions": mo.objective_directions,
            "objective_units": {
                "net_aep_mwh": "MWh/year",
                "lcoe_yuan_per_kwh": "CNY/kWh",
                "collection_length_m": "m",
            },
            "substation_xy": mo.substation_xy.tolist(),
            "n_solutions": len(solutions),
            "preference_weights": mo.preference_weights,
            "normalization": mo.normalization,
            "knee_solution_id": mo.knee_solution.solution_id,
            "knee_objectives": mo.knee_solution.objectives.as_dict(),
            "objective_extremes": extremes,
            "convergence_history": mo.convergence_history,
            "solutions_file": "pareto_front.json",
            "solutions_csv": "pareto_front.csv",
        }

    def _save_pareto_files(self, output_dir: str) -> None:
        """保存完整 Pareto 解集（JSON 含机位/网络明细，CSV 为目标表）。"""
        mo = self.multi_objective_result

        payload = {
            "algorithm": mo.algorithm,
            "seed": mo.seed,
            "objective_keys": list(mo.objective_keys),
            "objective_labels": mo.objective_labels,
            "objective_directions": mo.objective_directions,
            "preference_weights": mo.preference_weights,
            "normalization": mo.normalization,
            "substation_xy": mo.substation_xy.tolist(),
            "knee_solution_id": mo.knee_solution.solution_id,
            "solutions": [s.to_dict() for s in mo.solutions],
        }

        pareto_json = os.path.join(output_dir, "pareto_front.json")
        with open(pareto_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        pareto_csv = os.path.join(output_dir, "pareto_front.csv")
        with open(pareto_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "solution_id", "is_knee",
                "net_aep_mwh", "lcoe_yuan_per_kwh", "collection_length_m",
                "collection_cost_wanyuan", "total_capital_cost_wanyuan",
                "crowding_distance",
            ])
            knee_id = mo.knee_solution.solution_id
            for s in mo.solutions:
                d = s.objectives.as_dict()
                crowd_str = (
                    f"{s.crowding_distance:.6f}"
                    if np.isfinite(s.crowding_distance)
                    else ""
                )
                writer.writerow([
                    s.solution_id,
                    int(s.solution_id == knee_id),
                    f"{d['net_aep_mwh']:.6f}",
                    f"{d['lcoe_yuan_per_kwh']:.8f}",
                    f"{d['collection_length_m']:.6f}",
                    f"{s.objectives.collection_cost_wanyuan:.4f}",
                    f"{s.objectives.total_capital_cost_wanyuan:.4f}",
                    crowd_str,
                ])

    def run_full_analysis(
        self,
        run_baseline: bool = True,
        run_opt: bool = True,
        run_econ: bool = True,
        run_sweep: bool = False,
        run_viz: bool = True,
        save: bool = True,
    ) -> None:
        """运行完整分析流程。"""
        start_time = time.time()

        self._print_header("风电场机位布局优化分析")
        print(f"  风机: {self.config.turbine_model} x {self.config.n_turbines} 台")
        print(f"  尾流模型: {self.config.wake_model}")
        print(f"  平均风速: {self.wind_resource.overall_mean_speed:.2f} m/s")
        print(f"  场地面积: {self.boundary.area / 1e6:.2f} km²")
        print(f"  优化模式: {'多目标 NSGA-II（净AEP/LCOE/集电线路）' if self.config.multi_objective.enabled else f'单目标 {self.config.optimization.algorithm.upper()}（净AEP）'}")

        if run_baseline:
            self.run_baseline()

        if run_opt:
            if self.config.multi_objective.enabled:
                self.run_multi_objective_optimization()
            else:
                self.run_optimization()

        if run_econ:
            self.run_economic_analysis()

        if run_sweep:
            self.run_turbine_sweep(
                min_turbines=getattr(self, '_min_turbines', 5),
                max_turbines=getattr(self, '_max_turbines', 25),
            )

        if run_viz:
            self.run_visualization()

        if save:
            self.save_results()

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  全部分析完成! 耗时: {elapsed:.1f} 秒")
        print(f"{'='*60}\n")


def build_argparser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="风电场机位布局优化工具 - 尾流计算、布局优化、经济性评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认配置运行完整分析
  python -m wind_farm_opt

  # 从配置文件运行
  python -m wind_farm_opt --config my_config.json

  # 自定义参数运行
  python -m wind_farm_opt --n-turbines 20 --turbine V164-9.5MW --wake-model gaussian

  # 仅评估不优化
  python -m wind_farm_opt --no-optimization

  # 启用风机台数扫描
  python -m wind_farm_opt --sweep --min-turbines 10 --max-turbines 30
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径(JSON格式)",
    )

    parser.add_argument(
        "--n-turbines",
        type=int,
        default=None,
        help="风机台数",
    )

    parser.add_argument(
        "--turbine",
        type=str,
        default=None,
        choices=["V126-3.45MW", "V164-9.5MW"],
        help="风机型号",
    )

    parser.add_argument(
        "--wake-model",
        type=str,
        default=None,
        choices=["jensen", "gaussian"],
        help="尾流模型: jensen 或 gaussian",
    )

    parser.add_argument(
        "--wake-decay",
        type=float,
        default=None,
        help="尾流衰减系数 (Jensen模型)",
    )

    parser.add_argument(
        "--boundary",
        type=str,
        default=None,
        choices=["rectangular", "hexagonal", "irregular"],
        help="场地边界类型",
    )

    parser.add_argument(
        "--width",
        type=float,
        default=None,
        help="矩形场地宽度 (m)",
    )

    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="矩形场地高度 (m)",
    )

    parser.add_argument(
        "--min-spacing",
        type=float,
        default=None,
        help="最小间距倍数（转子直径倍数）",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["ga", "pso"],
        help="优化算法: ga(遗传算法) 或 pso(粒子群)",
    )

    parser.add_argument(
        "--population",
        type=int,
        default=None,
        help="种群/粒子群大小",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="最大迭代代数",
    )

    parser.add_argument(
        "--no-optimization",
        action="store_true",
        help="仅评估基线布局，不执行优化",
    )

    parser.add_argument(
        "--multi-objective",
        action="store_true",
        help="启用多目标模式：同时优化净AEP、LCOE与集电线路长度 (NSGA-II)",
    )

    parser.add_argument(
        "--mo-population",
        type=int,
        default=None,
        help="多目标模式种群大小",
    )

    parser.add_argument(
        "--mo-generations",
        type=int,
        default=None,
        help="多目标模式最大迭代代数",
    )

    parser.add_argument(
        "--archive-size",
        type=int,
        default=None,
        help="多目标 Pareto 外部存档容量上限",
    )

    parser.add_argument(
        "--substation-x",
        type=float,
        default=None,
        help="升压站X坐标 (m)，缺省取场地顶点质心",
    )

    parser.add_argument(
        "--substation-y",
        type=float,
        default=None,
        help="升压站Y坐标 (m)，缺省取场地顶点质心",
    )

    parser.add_argument(
        "--cable-cost",
        type=float,
        default=None,
        help="集电线路单位造价 (万元/km)",
    )

    parser.add_argument(
        "--prefer-aep",
        type=float,
        default=None,
        help="膝点选择中净AEP的归一化偏好权重（默认1.0）",
    )

    parser.add_argument(
        "--prefer-lcoe",
        type=float,
        default=None,
        help="膝点选择中LCOE的归一化偏好权重（默认1.0）",
    )

    parser.add_argument(
        "--prefer-cable",
        type=float,
        default=None,
        help="膝点选择中集电线路长度的归一化偏好权重（默认1.0）",
    )

    parser.add_argument(
        "--no-economic",
        action="store_true",
        help="跳过经济性分析",
    )

    parser.add_argument(
        "--sweep",
        action="store_true",
        help="启用风机台数扫描分析",
    )

    parser.add_argument(
        "--min-turbines",
        type=int,
        default=5,
        help="台数扫描最小值",
    )

    parser.add_argument(
        "--max-turbines",
        type=int,
        default=25,
        help="台数扫描最大值",
    )

    parser.add_argument(
        "--electricity-price",
        type=float,
        default=None,
        help="上网电价 (元/kWh)",
    )

    parser.add_argument(
        "--discount-rate",
        type=float,
        default=None,
        help="折现率 (0-1)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="不生成图表",
    )

    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="显示图表窗口",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子",
    )

    parser.add_argument(
        "--generate-config",
        type=str,
        default=None,
        help="生成示例配置文件并退出",
    )

    return parser


def main() -> int:
    """主函数入口。"""
    parser = build_argparser()
    args = parser.parse_args()

    if args.generate_config:
        config = create_sample_config()
        config.to_json(args.generate_config)
        print(f"示例配置已生成: {os.path.abspath(args.generate_config)}")
        return 0

    if args.config:
        config = WindFarmConfig.from_json(args.config)
    else:
        config = create_sample_config()

    if args.n_turbines is not None:
        config.n_turbines = args.n_turbines
    if args.turbine is not None:
        config.turbine_model = args.turbine
    if args.wake_model is not None:
        config.wake_model = args.wake_model
    if args.wake_decay is not None:
        config.wake_decay = args.wake_decay
    if args.boundary is not None:
        config.boundary_type = args.boundary
    if args.width is not None:
        config.boundary_params["width"] = args.width
    if args.height is not None:
        config.boundary_params["height"] = args.height
    if args.min_spacing is not None:
        config.optimization.min_spacing_multiple = args.min_spacing
    if args.algorithm is not None:
        config.optimization.algorithm = args.algorithm
    if args.population is not None:
        config.optimization.population_size = args.population
    if args.iterations is not None:
        config.optimization.max_iterations = args.iterations
    if args.seed is not None:
        config.optimization.seed = args.seed
    if args.electricity_price is not None:
        config.economic.electricity_price = args.electricity_price
    if args.discount_rate is not None:
        config.economic.discount_rate = args.discount_rate
    if args.output_dir is not None:
        config.visualization.save_dir = args.output_dir
    if args.no_plots:
        config.visualization.save_plots = False
    if args.show_plots:
        config.visualization.show_plots = True
    if args.no_economic:
        config.economic.enable_analysis = False

    if args.multi_objective:
        config.multi_objective.enabled = True
    if args.mo_population is not None:
        config.multi_objective.population_size = args.mo_population
    if args.mo_generations is not None:
        config.multi_objective.max_generations = args.mo_generations
    if args.archive_size is not None:
        config.multi_objective.archive_size = args.archive_size
    if args.substation_x is not None:
        config.multi_objective.substation_xy = [
            args.substation_x,
            args.substation_y if args.substation_y is not None else args.substation_x * 0.0,
        ]
    elif args.substation_y is not None:
        config.multi_objective.substation_xy = [0.0, args.substation_y]
    if args.cable_cost is not None:
        config.multi_objective.cable_cost_per_km_wanyuan = args.cable_cost
    if args.prefer_aep is not None:
        config.multi_objective.preference_weights["net_aep_mwh"] = args.prefer_aep
    if args.prefer_lcoe is not None:
        config.multi_objective.preference_weights["lcoe_yuan_per_kwh"] = args.prefer_lcoe
    if args.prefer_cable is not None:
        config.multi_objective.preference_weights["collection_length_m"] = args.prefer_cable

    cli = WindFarmOptimizerCLI(config)
    cli._min_turbines = args.min_turbines
    cli._max_turbines = args.max_turbines

    try:
        cli.run_full_analysis(
            run_baseline=True,
            run_opt=not args.no_optimization,
            run_econ=not args.no_economic,
            run_sweep=args.sweep,
            run_viz=not args.no_plots,
            save=True,
        )
        return 0
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
