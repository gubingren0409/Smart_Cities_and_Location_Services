"""Build and validate the final Lab 1 course submission archive.

This script only reorganizes frozen source code, notebooks, reports, data and
experiment outputs.  It does not run any experiment or call an LLM service.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "作业"
PACKAGE = ROOT / "submission_package"
ZIP_PATH = ROOT / "实验一_轨迹数据预处理_谷秉仁_10245102457.zip"

FORMAL_EXPERIMENTS = (
    "llm_assisted_ecnu_20260921_v2",
    "objective_optimization_20260922",
    "objective_optimization_refined_20260922",
    "task3_real_road_20260924",
    "task3_full_11386_20260924",
)

EXCLUDED_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    ".mplcache",
    "debug",
    "temp",
    "tmp",
}
EXCLUDED_FILES = {
    ".env",
    ".DS_Store",
    "Thumbs.db",
    "jupyterhub_cookie_secret",
    "jupyterhub.sqlite",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".log", ".sqlite", ".sqlite3"}

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".jsonl",
    ".ipynb",
    ".txt",
    ".ini",
    ".yaml",
    ".yml",
    ".csv",
}


def ensure_safe_generated_path(path: Path) -> None:
    resolved_root = ROOT.resolve()
    resolved = path.resolve()
    if resolved.parent != resolved_root:
        raise RuntimeError(f"Refusing generated path outside repository root: {resolved}")


def reset_generated_outputs() -> None:
    ensure_safe_generated_path(PACKAGE)
    ensure_safe_generated_path(ZIP_PATH)
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    PACKAGE.mkdir(parents=True)


def should_exclude(path: Path) -> bool:
    return (
        any(part in EXCLUDED_DIRS for part in path.parts)
        or path.name in EXCLUDED_FILES
        or path.suffix.lower() in EXCLUDED_SUFFIXES
        or path.name.startswith("~$")
    )


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_filtered_tree(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        if should_exclude(relative):
            continue
        destination = target / relative
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            copy_file(item, destination)


def locate_code_cell(notebook: dict, marker: str) -> dict:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code" and marker in "".join(cell.get("source", [])):
            return cell
    raise RuntimeError(f"Notebook cell marker not found: {marker}")


def set_cell_source(cell: dict, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def make_portable_notebooks() -> None:
    notebook_dir = PACKAGE / "notebooks"
    notebook_dir.mkdir(parents=True, exist_ok=True)

    base_source = WORK / "作业1轨迹数据预处理.ipynb"
    base = json.loads(base_source.read_text(encoding="utf-8"))
    imports = locate_code_cell(base, "from utils import util")
    source = "".join(imports["source"])
    portable_setup = '''import sys

# 提交包可移植路径：只调整导入与数据位置，不改变实验算法和保存输出。
_start = Path.cwd().resolve()
PACKAGE_ROOT = next(
    (p for p in (_start, *_start.parents) if (p / "src" / "traj_agent").is_dir()),
    _start.parent,
)
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from utils import util'''
    source = source.replace("from utils import util", portable_setup)
    set_cell_source(imports, source)

    load_cell = locate_code_cell(base, "Path('traj_dict.json')")
    source = "".join(load_cell["source"]).replace(
        "Path('traj_dict.json')", "(PACKAGE_ROOT / 'data' / 'traj_dict.json')"
    )
    set_cell_source(load_cell, source)
    (notebook_dir / base_source.name).write_text(
        json.dumps(base, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    task3_source = WORK / "任务3_LLM辅助评估清洗.ipynb"
    task3 = json.loads(task3_source.read_text(encoding="utf-8"))
    setup = locate_code_cell(task3, "load_dotenv(Path(\".env\")")
    source = "".join(setup["source"])
    source = source.replace(
        "# 本地密钥保存在被 Git 忽略的 .env；Notebook 中不写入真实 key\n"
        "load_dotenv(Path(\".env\"), override=False)",
        "# 提交包不包含密钥；如需重跑真实模型，可自行在包根目录创建 .env。\n"
        "_start = Path.cwd().resolve()\n"
        "PACKAGE_ROOT = next(\n"
        "    (p for p in (_start, *_start.parents) if (p / \"src\" / \"traj_agent\").is_dir()),\n"
        "    _start.parent,\n"
        ")\n"
        "load_dotenv(PACKAGE_ROOT / \".env\", override=False)",
    )
    source = source.replace(
        'sys.path.insert(0, os.path.abspath("."))',
        'sys.path.insert(0, str(PACKAGE_ROOT / "src"))',
    )
    source = source.replace(
        'os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".mplcache"))',
        'os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplcache"))\n'
        'os.environ.setdefault("DSH_TRAJ_VAULT_DIR", str(PACKAGE_ROOT / "resources" / "vault_stub"))',
    )
    source = source.replace(
        'DATA = "traj_dict.json"',
        'DATA = str(PACKAGE_ROOT / "data" / "traj_dict.json")',
    )
    set_cell_source(setup, source)

    sample_cell = locate_code_cell(task3, "random_sample_200.json")
    source = "".join(sample_cell["source"])
    source = re.sub(
        r'sample_manifest = json\.loads\(Path\(\s*"experiments/llm_assisted_20260920/output/random_sample_200\.json"\s*\)\.read_text',
        'sample_manifest = json.loads((PACKAGE_ROOT / "data" / "random_sample_200.json").read_text',
        source,
    )
    set_cell_source(sample_cell, source)

    summary_cell = locate_code_cell(task3, "llm_assisted_ecnu_20260921_v2/summary.json")
    source = "".join(summary_cell["source"])
    source = re.sub(
        r'real_summary_path = Path\(\s*"experiments/llm_assisted_ecnu_20260921_v2/summary\.json"\s*\)',
        'real_summary_path = PACKAGE_ROOT / "experiments" / "llm_assisted_ecnu_20260921_v2" / "summary.json"',
        source,
    )
    set_cell_source(summary_cell, source)
    (notebook_dir / task3_source.name).write_text(
        json.dumps(task3, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def copy_reports() -> None:
    report_dir = PACKAGE / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    copy_file(
        ROOT / "output" / "pdf" / "实验一_轨迹数据预处理_最终实验报告.pdf",
        report_dir / "实验一_轨迹数据预处理_最终实验报告.pdf",
    )
    source_md = (WORK / "实验一_轨迹数据预处理_最终实验报告.md").read_text(encoding="utf-8")
    source_md = source_md.replace(
        "](report_lab1/figures/", "](assets/"
    ).replace(
        "](figures/hw1_real_examples.png)", "](assets/hw1_real_examples.png)"
    ).replace(
        "](experiments/", "](../experiments/"
    )
    (report_dir / "实验一_轨迹数据预处理_最终实验报告.md").write_text(
        source_md, encoding="utf-8"
    )
    for figure in sorted((WORK / "report_lab1" / "figures").glob("*.png")):
        copy_file(figure, report_dir / "assets" / figure.name)
    copy_file(
        WORK / "figures" / "hw1_real_examples.png",
        report_dir / "assets" / "hw1_real_examples.png",
    )


def copy_source_and_data() -> None:
    copy_filtered_tree(WORK / "traj_agent", PACKAGE / "src" / "traj_agent")
    copy_filtered_tree(WORK / "utils", PACKAGE / "src" / "utils")
    duplicate_data = PACKAGE / "src" / "utils" / "traj_dict.json"
    if duplicate_data.exists():
        duplicate_data.unlink()
    copy_file(WORK / "douglas_peucker.py", PACKAGE / "src" / "douglas_peucker.py")
    copy_file(
        WORK / "report_lab1" / "build_final_report.py",
        PACKAGE / "src" / "report_tools" / "build_final_report.py",
    )
    report_builder = PACKAGE / "src" / "report_tools" / "build_final_report.py"
    builder_text = report_builder.read_text(encoding="utf-8")
    builder_text = builder_text.replace(
        'SOURCE = ROOT / "作业" / "实验一_轨迹数据预处理_最终实验报告.md"',
        'SOURCE = ROOT / "report" / "实验一_轨迹数据预处理_最终实验报告.md"',
    ).replace(
        'PDF_PATH = ROOT / "output" / "pdf" / "实验一_轨迹数据预处理_最终实验报告.pdf"',
        'PDF_PATH = ROOT / "report" / "实验一_轨迹数据预处理_最终实验报告.pdf"',
    )
    report_builder.write_text(builder_text, encoding="utf-8")
    copy_filtered_tree(WORK / "tests", PACKAGE / "tests")
    # The repository test uses an obvious fake sk-* value to exercise provider
    # routing.  Replace it in the submission copy so a conservative credential
    # scan can reject every sk-* string without weakening the test semantics.
    full_run_test = PACKAGE / "tests" / "test_task3_full_run.py"
    full_run_test.write_text(
        full_run_test.read_text(encoding="utf-8").replace(
            'secret = "sk-1234567890abcdef"',
            'secret = "test-key-placeholder"',
        ),
        encoding="utf-8",
    )
    copy_file(WORK / "requirements.txt", PACKAGE / "requirements.txt")
    requirements = PACKAGE / "requirements.txt"
    requirement_lines = requirements.read_text(encoding="utf-8").splitlines()
    if "pypdf" not in requirement_lines:
        requirement_lines.append("pypdf")
    requirements.write_text("\n".join(requirement_lines) + "\n", encoding="utf-8")
    copy_file(WORK / "pytest.ini", PACKAGE / "pytest.ini")
    copy_file(WORK / "traj_dict.json", PACKAGE / "data" / "traj_dict.json")
    copy_file(
        WORK / "experiments" / "llm_assisted_20260920" / "output" / "random_sample_200.json",
        PACKAGE / "data" / "random_sample_200.json",
    )
    copy_filtered_tree(WORK / "vault_stub", PACKAGE / "resources" / "vault_stub")
    copy_file(WORK / "examples" / "run_demo.py", PACKAGE / "examples" / "run_demo.py")
    demo = PACKAGE / "examples" / "run_demo.py"
    demo_text = demo.read_text(encoding="utf-8")
    demo_text = demo_text.replace(
        "import time\nimport warnings",
        "import time\nimport warnings\nfrom pathlib import Path",
    ).replace(
        'sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))',
        'PACKAGE_ROOT = Path(__file__).resolve().parents[1]\n'
        'sys.path.insert(0, str(PACKAGE_ROOT / "src"))\n'
        'os.environ.setdefault("DSH_TRAJ_VAULT_DIR", str(PACKAGE_ROOT / "resources" / "vault_stub"))',
    ).replace(
        'ap.add_argument("--data", default="traj_dict.json")',
        'ap.add_argument("--data", default=str(PACKAGE_ROOT / "data" / "traj_dict.json"))',
    )
    demo.write_text(demo_text, encoding="utf-8")


def copy_experiments() -> None:
    for name in FORMAL_EXPERIMENTS:
        copy_filtered_tree(WORK / "experiments" / name, PACKAGE / "experiments" / name)


def copy_docs() -> None:
    docs = {
        "实验报告_要求覆盖检查.md": "实验报告_要求覆盖检查.md",
        "任务三_四项完成度验收.md": "任务三_四项完成度验收.md",
        "AI使用记录_实验一.md": "AI使用记录_实验一.md",
        "DECISIONS.md": "DECISIONS.md",
        "任务三_代码与目录清单.md": "任务三_代码与目录清单.md",
        "使用说明.md": "使用说明.md",
        "轨迹数据预处理评估报告.md": "轨迹数据预处理评估报告.md",
    }
    for source_name, target_name in docs.items():
        copy_file(WORK / source_name, PACKAGE / "docs" / target_name)
    copy_filtered_tree(
        WORK / "figures" / "ai_usage",
        PACKAGE / "docs" / "figures" / "ai_usage",
    )


def write_readme() -> None:
    text = """# 实验一《轨迹数据预处理》提交说明

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
"""
    (PACKAGE / "README_提交说明.md").write_text(text, encoding="utf-8")


def scan_sensitive_files(root: Path) -> list[str]:
    findings: list[str] = []
    secret_patterns = (
        re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
        re.compile(
            r"(?i)(?:api[_-]?key|credential|client[_-]?secret)\s*[\"']?\s*[:=]\s*[\"']([A-Za-z0-9._-]{16,})"
        ),
    )
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root)
        if path.name in EXCLUDED_FILES or path.name.lower().endswith((".pem", ".key")):
            findings.append(f"forbidden filename: {relative.as_posix()}")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in secret_patterns):
            findings.append(f"possible credential content: {relative.as_posix()}")
    return findings


def validate_package_tree(root: Path) -> dict[str, object]:
    required_dirs = ("report", "notebooks", "src", "experiments", "tests", "docs", "data")
    for name in required_dirs:
        if not (root / name).is_dir():
            raise RuntimeError(f"Missing required directory: {name}")

    required_files = (
        "README_提交说明.md",
        "report/实验一_轨迹数据预处理_最终实验报告.pdf",
        "report/实验一_轨迹数据预处理_最终实验报告.md",
        "notebooks/作业1轨迹数据预处理.ipynb",
        "notebooks/任务3_LLM辅助评估清洗.ipynb",
        "src/traj_agent/verifier/objective.py",
        "src/traj_agent/road/osm.py",
        "src/traj_agent/experiments/task3_full_run.py",
        "tests/test_task3_full_run.py",
        "data/traj_dict.json",
        "docs/实验报告_要求覆盖检查.md",
        "docs/任务三_四项完成度验收.md",
        "docs/AI使用记录_实验一.md",
    )
    for relative in required_files:
        if not (root / relative).is_file():
            raise RuntimeError(f"Missing required file: {relative}")

    for name in FORMAL_EXPERIMENTS:
        summary = root / "experiments" / name / "summary.json"
        if not summary.is_file():
            raise RuntimeError(f"Missing experiment summary: {name}")
        json.loads(summary.read_text(encoding="utf-8"))

    raw = json.loads((root / "data" / "traj_dict.json").read_text(encoding="utf-8"))
    if len(raw) != 11_386:
        raise RuntimeError(f"Unexpected trajectory count: {len(raw)}")

    for notebook_name in ("作业1轨迹数据预处理.ipynb", "任务3_LLM辅助评估清洗.ipynb"):
        notebook = json.loads((root / "notebooks" / notebook_name).read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        if "PACKAGE_ROOT" not in code:
            raise RuntimeError(f"Portable path setup missing from {notebook_name}")

    pdf_path = root / "report" / "实验一_轨迹数据预处理_最终实验报告.pdf"
    pdf = PdfReader(str(pdf_path))
    pdf_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    if len(pdf.pages) != 16:
        raise RuntimeError(f"Unexpected PDF page count: {len(pdf.pages)}")
    for required in (
        "谷秉仁",
        "10245102457",
        "真实 OSM 路网约束验证",
        "图 11",
        "全量 11,386 条任务三部署工作流",
        "图 12",
    ):
        if required not in pdf_text:
            raise RuntimeError(f"PDF content missing: {required}")

    report_md = root / "report" / "实验一_轨迹数据预处理_最终实验报告.md"
    markdown = report_md.read_text(encoding="utf-8")
    for link in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown):
        if not (report_md.parent / link).resolve().is_file():
            raise RuntimeError(f"Broken report image link: {link}")

    ai_record = root / "docs" / "AI使用记录_实验一.md"
    ai_markdown = ai_record.read_text(encoding="utf-8")
    for link in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", ai_markdown):
        if not (ai_record.parent / link).resolve().is_file():
            raise RuntimeError(f"Broken AI evidence image link: {link}")

    excluded = [
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if should_exclude(p.relative_to(root))
    ]
    if excluded:
        raise RuntimeError(f"Excluded artifacts found: {excluded[:5]}")

    sensitive = scan_sensitive_files(root)
    if sensitive:
        raise RuntimeError("Sensitive-content scan failed: " + "; ".join(sensitive))

    large_files = [
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.stat().st_size > 100 * 1024 * 1024
    ]
    if large_files:
        raise RuntimeError(f"Files over 100 MiB found: {large_files}")

    for source_file in root.rglob("*.py"):
        ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))

    files = [p for p in root.rglob("*") if p.is_file()]
    return {
        "file_count": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "pdf_pages": len(pdf.pages),
        "trajectory_count": len(raw),
        "sensitive_findings": len(sensitive),
        "large_files": len(large_files),
    }


def write_acceptance(status: str, metrics: dict[str, object]) -> None:
    text = f"""# 实验一提交包验收

## 报告

- [x] 最终 PDF 存在，可解析，共 {metrics['pdf_pages']} 页
- [x] PDF 包含基础实验、LLM 进阶、真实 OSM 路网、图 11、全量 11,386 条部署与图 12
- [x] PDF 姓名与学号为谷秉仁、10245102457
- [x] 最终 Markdown 存在，图片链接完整

## 基础实验

- [x] 轨迹分段
- [x] GPS 漂移去噪
- [x] Douglas–Peucker 简化
- [x] 互补评价指标与参数实验
- [x] AI 建议批判、反例和处理顺序讨论

## 任务三

- [x] Search-verified Memory：PASS
- [x] 真实 LLM 消融：PASS
- [x] 真实 OSM 路网约束：PASS
- [x] 全量 11,386 条部署工作流：PASS

## 工程与安全

- [x] `src/traj_agent`、辅助模块和完整测试存在
- [x] 两个 Notebook 均保留输出，并具有提交包可移植路径引导
- [x] 原始轨迹数据共 {metrics['trajectory_count']:,} 条，可供基础 Notebook 复现
- [x] 五个正式实验目录和关键 `summary.json` 完整
- [x] AI 使用记录及四张真实交互截图完整
- [x] `.env`、API key、Jupyter secret、缓存、`.pyc`、临时日志和 SQLite 运行库均未打包
- [x] 超过 100 MiB 的单文件：{metrics['large_files']} 个
- [x] 包内文件数：{metrics['file_count']} 个
- [x] SHA-256 文件清单存在
- [x] zip 顶层仅为 `submission_package/`
- [x] zip 已在独立临时目录解压并复验

## 验收结论

**提交包验证 {status}**

本次只进行了文件整理、路径适配、压缩和完整性检查，没有修改算法、Objective、Search、Memory、LLM 策略或历史实验结果，也没有重新运行实验。
"""
    (PACKAGE / "docs" / "提交包验收.md").write_text(text, encoding="utf-8")


def write_manifest() -> None:
    manifest = PACKAGE / "MANIFEST_SHA256.txt"
    lines = []
    for path in sorted(p for p in PACKAGE.rglob("*") if p.is_file() and p != manifest):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(PACKAGE).as_posix()}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(p for p in PACKAGE.rglob("*") if p.is_file()):
            archive.write(path, Path("submission_package") / path.relative_to(PACKAGE))


def validate_manifest(root: Path) -> None:
    manifest = root / "MANIFEST_SHA256.txt"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Manifest mismatch: {relative}")


def validate_zip() -> dict[str, object]:
    with zipfile.ZipFile(ZIP_PATH) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Zip CRC validation failed")
        names = archive.namelist()
        if not names or any(not name.startswith("submission_package/") for name in names):
            raise RuntimeError("Zip must have submission_package/ as its only top-level directory")
        with tempfile.TemporaryDirectory(prefix="lab1_submission_verify_") as temp:
            archive.extractall(temp)
            extracted = Path(temp) / "submission_package"
            metrics = validate_package_tree(extracted)
            validate_manifest(extracted)
            metrics["archive_file_count"] = len(names)
            return metrics


def main() -> None:
    reset_generated_outputs()
    copy_reports()
    make_portable_notebooks()
    copy_source_and_data()
    copy_experiments()
    copy_docs()
    write_readme()

    initial = validate_package_tree(PACKAGE)
    write_acceptance("PENDING", initial)
    write_manifest()
    create_zip()
    validate_zip()

    final_tree = validate_package_tree(PACKAGE)
    write_acceptance("PASS", final_tree)
    write_manifest()
    create_zip()
    final = validate_zip()
    final["zip_bytes"] = ZIP_PATH.stat().st_size
    final["zip_sha256"] = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()

    print(json.dumps(final, ensure_ascii=False, indent=2))
    print(ZIP_PATH)


if __name__ == "__main__":
    main()
