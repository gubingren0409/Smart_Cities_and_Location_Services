# 智慧城市与位置服务：Lab 1 轨迹数据预处理

本分支保存实验一的课程材料、可复现 Notebook、数据、代码、图表和 AI 使用记录。

## 主要内容

- `实验课1.pptx`：老师实验课材料。
- `作业/作业1轨迹数据预处理.ipynb`：分段、去噪、Douglas–Peucker 简化、参数实验、真实轨迹可视化与 AI 建议反例。
- `作业/轨迹数据预处理评估报告.docx`、`.pdf`：根据 Notebook 实际运行结果整理的正式实验报告；Markdown 源稿与可复现制图脚本一并保留。
- `作业/AI使用记录_实验一.md`：真实 Human–AI 交互证据及修改前后位置说明。
- `作业/traj_dict.json`：实验数据。
- `作业/utils/`、`作业/douglas_peucker.py`：老师框架及基础工具。
- `作业/任务3_LLM辅助评估清洗.ipynb`、`作业/traj_agent/`：课程选做部分的辅助评估框架与演示。
- `作业/任务三_代码与目录清单.md`：任务三源码、测试、实验结果和展示材料的完整导航。
- `作业/tests/`：几何、清洗、简化、代理、核验和记忆模块测试。
- `作业/figures/`、`作业/experiments/`：真实运行生成的图表和实验结果。

## 运行方式

```powershell
cd 作业
jupyter lab 作业1轨迹数据预处理.ipynb
```

运行测试：

```powershell
cd 作业
python -m pytest tests -q
```

运行辅助评估演示：

```powershell
cd 作业
python examples/run_demo.py --sample 200
```

未设置模型 API 凭据时，辅助评估演示使用项目自带的 `MockProvider`。仓库不包含任何真实 API 密钥。

## 项目约定

长期工作流见 [PROJECT_WORKFLOW.md](PROJECT_WORKFLOW.md)，代理入口见 [AGENTS.md](AGENTS.md)。
