# HW2：Gensim 表示学习与 FastText 向量文本分类

对应《大语言模型与应用实践》L2.part1 P137；小规模 CPU 实验。

**[完整实验报告（可复制到 Piazza）](docs/experiment_report.md)**

| 任务 | 表示 | 准确率 | 宏平均 F1 |
|---|---|---:|---:|
| 二分类，30训练／10测试 | TF-IDF | 1.0000 | 1.0000 |
| 二分类，30训练／10测试 | FastText平均向量 | 0.9000 | 0.8990 |
| 四分类，60训练／20测试 | TF-IDF | 0.9000 | 0.9000 |
| 四分类，60训练／20测试 | FastText平均向量 | 0.8000 | 0.7833 |

Gensim FastText 用于学习词向量，分类配合外部逻辑回归；不是官方 fastText 监督分类训练。样本经过选择，标签有 AI 辅助并由学习者采纳，测试很小，结果不代表一般性能。

## 发布内容

代码、锁定依赖、主题标注元数据、去除全文和本机路径的原始实验记录。**不提供原始语料、处理后全文、训练模型或 `.venv`。**标注中的正文 SHA256 由 `scripts/corpus_io.py` 配合本地语料验证，恢复正文后再运行实验。

`records/` 是历史成绩快照；重新运行的输出在被忽略的 `results/`，不会覆盖历史快照。历史 SHA256 指向原本地运行输入，公开标注经过删除全文的变换，因此文件哈希不同；公开版也已单独端到端验证了相同划分、逐篇预测和指标，见 [检查记录](records/public_reproduction_check.json)。

## 环境

本机实测 Windows、Python 3.12.3；Gensim 4.4.0、scikit-learn 1.9.1、jieba 0.42.1；完整版本在 `requirements.lock.txt`。同环境已复现，未验证所有其他操作系统／版本。

```powershell
git clone https://github.com/lhjbadrain/llmp-hw2-gensim-fasttext.git
cd llmp-hw2-gensim-fasttext
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```

## 准备人民日报输入

请按课程或数据来源要求自行取得一月带音原文件，放到：

```text
data/raw/1998-01-2003版-带音.txt
```

本实验原始文件 SHA256：

```text
37b85555df2f72642ff9815ac01f51d4f50024d80ca0e2e8a6be6f93b1038639
```

清洗程序会严格检查版本。不使用不同来源文件硬凑本报告结果。外部 CNews 文件由下载程序取得，固定版本、约11.8MB，并检查SHA256；已有正确文件直接复用。

## 从空输出目录运行

以下按 PowerShell 编写：

```powershell
$env:PYTHONHASHSEED = "42"
$env:PYTHONIOENCODING = "utf-8"
$hw2python = ".\.venv\Scripts\python.exe"
& $hw2python scripts/00_clean_january.py
& $hw2python scripts/00_download_external.py
& $hw2python scripts/03_prepare_external_v3.py
& $hw2python scripts/04_bow_tfidf.py
& $hw2python scripts/05_word2vec.py
& $hw2python scripts/06_fasttext.py
& $hw2python scripts/07_document_vectors.py
& $hw2python scripts/08_prepare_binary_split.py
& $hw2python scripts/09_train_compare.py
& $hw2python scripts/10_prepare_fourclass_split.py
& $hw2python scripts/09_train_compare.py --task fourclass
```

程序保护已有输出，失败时先检查原因。已有完整运行后，若需要另一份空目录，可用 `11_make_repro_copy.py --name reproduce_run_2`，它不复制旧模型和成绩。副本用同一个已安装依赖的 Python 执行。

## 文件说明

- `scripts/00*`：原始一月清洗、外部新闻取得；`reference/` 中复用前一次作业的清洗函数。
- `scripts/01*`、`02*`：教学检查和标注辅助；最终标签已采纳，核心复现不需重做交互标注。
- `scripts/03_prepare_external_v3.py`：正式600篇独立语料。原版及v2保留清洗迭代代码，**正式流程只运行v3**。
- `scripts/04*`—`07*`：Gensim表示与词向量／文章向量实验。
- `scripts/08*`—`10*`：数据划分与统一分类评估。
- `data/annotations/`：标签、理由、来源ID、正文校验值。`其他`是二分类之外的粗标签，四分类另保存细分，不改变旧任务。
- `records/`：原始实测数值、预测标签、参数、边界说明和复现证据，全文已移除。

只用正文词生成特征，编号／版面／标签／理由均不参与。TF-IDF仅拟合训练集；FastText冻结自外部语料。四分类复用二分类的划分归属，不是独立重复实验。未另设验证集，未按测试成绩调参。

更详细的数据来源、标注方法、全部错例、论文源码对照、耗时口径、AI协作与局限见[实验报告](docs/experiment_report.md)。
