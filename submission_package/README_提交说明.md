# 实验一《轨迹数据预处理》提交说明

## 一、提交信息

| 项目 | 内容 |
|---|---|
| 课程 | 智慧城市与位置服务 |
| 实验 | 实验一：轨迹数据预处理 |
| 姓名 | 谷秉仁 |
| 学号 | 10245102457 |
| 数据规模 | 11,386 条车辆轨迹，1,173,410 个 GPS 点 |

## 二、完成内容

基础实验完成轨迹分段、GPS 漂移去噪、Douglas–Peucker 简化、互补评价指标、参数敏感性、真实轨迹可视化、AI 建议反例与处理顺序讨论。

任务三沿用助教框架完成真实模型消融、Search-verified Memory、真实 OSM 路网约束和 11,386 条全量部署工作流。正式完成度如下：

| 项目 | 状态 |
|---|---|
| Memory 增益 | PASS |
| 真实 LLM 消融 | PASS |
| OSM 路网约束 | PASS |
| 全量 11,386 条部署工作流 | PASS |

## 三、目录说明

- `report/`：最终 PDF、Markdown 及 Markdown 图片资源。
- `notebooks/`：基础实验和任务三 Notebook；仅增加提交包相对路径引导，算法、已有输出和实验结果未改动。
- `src/`：`traj_agent` 完整源码，以及基础 Notebook 使用的 `utils`、DP helper 和 PDF 构建脚本。
- `experiments/`：五个正式实验目录及其配置、JSON/JSONL、图表和总结。
- `tests/`：完整回归测试源码。
- `data/`：原始轨迹数据和 Notebook 固定抽样清单。原始数据是基础 Notebook 从头运行的必要输入。
- `docs/`：要求覆盖检查、四项验收、AI 使用记录及截图、设计决策和代码目录说明。
- `resources/`：任务三使用的只读知识库示例。
- `MANIFEST_SHA256.txt`：包内文件完整性校验值。

## 四、查看与复现

```powershell
python -m pip install -r requirements.txt
jupyter lab
```

打开 `notebooks/作业1轨迹数据预处理.ipynb` 可查看并复现基础实验；打开 `notebooks/任务3_LLM辅助评估清洗.ipynb` 可查看任务三流程。Notebook 的路径引导会自动定位包内 `src/`、`data/` 和 `experiments/`。

离线回归测试：

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests -q
```

保存的正式结果无需 API key 即可查看。提交包不包含任何密钥；若自行重跑真实 LLM 实验，需要在本机配置 `DSH_TRAJ_LLM_API_KEY`，不得把密钥写回提交包。

## 五、证据边界

真实 OSM 道路网络已经用于固定 holdout 验证，但仍缺少人工标注的车辆真实行驶道路路径真值；全量部署完成覆盖、路由、成本和恢复验证，未逐车运行 strong reference。主要收益来自 Search-verified Episodic Memory 与确定性搜索，LLM 主要承担参数区域提议。
