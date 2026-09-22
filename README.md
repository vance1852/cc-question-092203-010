# 风电场布局优化工具

这个项目用于估算风电场的年发电量，并比较不同风机布局和尾流模型的结果。项目包含风机与风资源模型、场地边界和间距约束、遗传算法与粒子群优化、经济性分析以及无界面图表输出。

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

## 多目标模式

默认情况下仍运行单目标 GA/PSO（最大化净 AEP），输出结构保持不变。加入
`--multi-objective` 后切换为 NSGA-II 多目标优化，同时比较三个原始量纲目标：

| 目标 | 键名 | 量纲 | 方向 |
| --- | --- | --- | --- |
| 净年发电量 | `net_aep_mwh` | MWh/年 | 越大越好 |
| 度电成本 | `lcoe_yuan_per_kwh` | 元/kWh（初始投资含集电系统） | 越小越好 |
| 集电线路长度 | `collection_length_m` | m（机位+升压站 MST 网络） | 越小越好 |

```bash
# 启用多目标，输出完整 Pareto 解集
python -m wind_farm_opt --multi-objective \
    --n-turbines 15 --mo-population 50 --mo-generations 80 --archive-size 80 \
    --substation-x 0 --substation-y 0 --cable-cost 35 \
    --prefer-aep 2 --prefer-lcoe 1 --prefer-cable 1 \
    --output-dir output_mo
```

- **非支配解集与多样性**：算法维护受场地边界与最小间距约束的可行非支配
  解集，按非支配层级 + 拥挤度选择，并以外部存档保多样性；相同 `--seed`
  重复运行，方案编号（P01、P02…，按净AEP降序的确定性次序）与目标值
  完全一致。
- **膝点方案**：在 Pareto 观测范围内做 min-max 归一化后，按
  `--prefer-aep/--prefer-lcoe/--prefer-cable`（配置文件
  `multi_objective.preference_weights`）加权选择膝点；权重无需预先归一化。
- **结果回溯**：
  - `pareto_front.json` —— 每个方案的机位坐标、三个目标原值、升压站
    坐标与 MST 电缆边（节点、长度、是否接升压站）；
  - `pareto_front.csv` —— 目标汇总表（含膝点标记）；
  - `results.json` 的 `multi_objective` 段 —— 归一化区间、偏好权重、
    各目标理想/最差点、存档演化历史；
  - 图表 `pareto_pairwise.png`、`pareto_parallel.png`、
    `pareto_convergence.png`、`knee_collection_network.png`，
    方案编号与结果文件一一对应。

也可以在配置文件中通过 `multi_objective` 段启用（`enabled: true`）。
未启用该段时，单目标 GA、PSO 的行为、图表与 `results.json` 结构完全兼容。
