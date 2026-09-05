# upstream_data/ —— vendored 上游数据集

`verify_provenance.py` 用这六个文件核验训练数据的每一行是否**逐字出自公开数据集**。
收进本仓库的目的：让溯源校验（`PROV=1`）**无需访问 HuggingFace** 即可运行。
文件内容与上游逐字节一致，未做任何修改。

抓取日期 **2026-09-05**，合计约 500 KB。

| 本地文件 | HuggingFace 来源 | 上游路径 | 许可证 |
|---|---|---|---|
| `crux.jsonl` | `cruxeval-org/cruxeval` | `test.jsonl` | MIT |
| `lcb.parquet` | `livecodebench/execution-v2` | `data/test-00000-of-00001.parquet` | MIT |
| `mbpp_train.parquet` | `google-research-datasets/mbpp` | `full/train-00000-of-00001.parquet` | CC-BY-4.0 |
| `mbpp_test.parquet` | 同上 | `full/test-00000-of-00001.parquet` | CC-BY-4.0 |
| `mbpp_val.parquet` | 同上 | `full/validation-00000-of-00001.parquet` | CC-BY-4.0 |
| `mbpp_prompt.parquet` | 同上 | `full/prompt-00000-of-00001.parquet` | CC-BY-4.0 |

## 关于 LiveCodeBench 的许可证

HuggingFace 数据集页面只标了笼统的 `cc`（没说是 BY / NC / ND 哪种）。
查证其官方仓库 https://github.com/LiveCodeBench/LiveCodeBench ，
许可证为 **MIT**，允许附出处再分发。

## 论文引用

- **CRUXEval** — Gu et al. 2024, arXiv:2401.03065
- **LiveCodeBench** — Jain et al. 2024, arXiv:2403.07974
- **MBPP** — Austin et al. 2021, arXiv:2108.07732

## 怎么更新

```bash
cd upstream_data
curl -L -o <本地名> "https://huggingface.co/datasets/<repo>/resolve/main/<上游路径>"
```
文件名必须与 `verify_provenance.py` 的 `FILES` 表第三列一致，否则找不到。
