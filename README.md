# 风电场布局优化工具

这个项目用于估算风电场的年发电量，并比较不同风机布局和尾流模型的结果。项目包含风机与风资源模型、场地边界和间距约束、遗传算法与粒子群优化、经济性分析、多目标 Pareto 优化以及无界面图表输出。

## 安装

建议使用 Python 3.10 或更新版本，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 可以使用 `.venv\\Scripts\\Activate.ps1` 激活环境。

## 快速验证

```bash
python quick_test.py
```

快速验证会覆盖模型、约束、年发电量、优化、经济性和图表生成，并在 `test_output/` 写入临时图片。该目录不会纳入版本控制。

## 完整分析

```bash
python -m wind_farm_opt --help
python -m wind_farm_opt --n-turbines 15 --iterations 100 --population 50 --output-dir output
```

也可以先生成配置文件，再通过 `--config` 运行：

```bash
python -m wind_farm_opt --generate-config my_config.json
python -m wind_farm_opt --config my_config.json
```

所有运行结果默认写入 `output/`，可以用 `--no-plots` 跳过图表生成。命令行使用无界面绘图后端，适合容器和服务器环境。

## 多目标 Pareto 优化

默认运行仍是单目标（最大化净 AEP）的 GA / PSO，行为与输出结构保持不变。加 `--multi-objective` 后切换为三目标模式，同时比较：

| 目标 | 键名 | 量纲 | 方向 |
| --- | --- | --- | --- |
| 净年发电量 | `net_aep_mwh` / `net_aep_gwh` | MWh / GWh 每年 | 越大越好 |
| 度电成本 | `lcoe_yuan_per_kwh` | 元/kWh（初始投资计入集电线路造价） | 越小越好 |
| 集电线路长度 | `cable_length_m` / `cable_length_km` | m / km | 越小越好 |

集电线路长度按“机位—升压站”连通网络的最小生成树（MST）估算；升压站默认取各布局机位重心，也可用 `--substation x,y` 固定。搜索始终在满足场地边界与最小间距约束的可行解中进行，采用非支配排序 + 拥挤距离（NSGA-II，或多目标粒子群 `mopso`）维护分布均匀的 Pareto 非支配解集。

```bash
# NSGA-II（默认）
python -m wind_farm_opt --multi-objective --n-turbines 15 \
    --mo-population 40 --mo-iterations 80 --output-dir output_mo

# 多目标粒子群
python -m wind_farm_opt --multi-objective --mo-algorithm mopso \
    --mo-population 40 --mo-iterations 100

# 偏重发电量的膝点偏好（aep,lcoe,cable 相对权重，无需归一）
python -m wind_farm_opt --multi-objective --knee-weights 2,1,1

# 固定升压站位置并指定线缆造价
python -m wind_farm_opt --multi-objective --substation 200,100 --cable-cost 80
```

输出结果（`results.json` 的 `multi_objective` 段，图形 `pareto_front.png`、`pareto_parallel.png`、`pareto_convergence.png`、`knee_network.png`）：

- `solutions`：完整 Pareto 解集，每个方案含稳定编号（`P00`、`P01`……）、机位坐标 `positions_m`、三目标的**原始量纲**值、归一化值、偏好加权得分，以及可逐段回溯的集电网络边列表；
- `knee_solution_id` / `is_knee`：按显式归一化偏好权重（`knee_weights_normalized`）从 Pareto 前沿选出的膝点（折中）方案；
- 解按净 AEP 降序确定序编号，相同 `--seed` 可逐位复现排序、机位与目标值。

膝点方案同时作为主“优化后布局”参与既有经济性分析与对比图，便于评审会上在同一口径下查看取舍。
