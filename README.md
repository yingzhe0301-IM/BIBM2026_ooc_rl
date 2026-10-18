# A Conservative Certification Protocol for Treatment De-escalation in Patient-Derived Organoids

Yingzhe Qin, Yu Shrike Zhang, and Ming Shao. IEEE BIBM 2026 Workshop on Artificial Intelligence for Precision Medicine (AI4PM).

A lower-burden treatment rule may replace a higher-burden baseline when its loss in final tumor killing stays within an allowance that grows with the burden it saves. The protocol certifies such a rule at a shared decision point of a patient-derived organoid screen in three stages: a diagnostic of what the decision-point state supports, a ranking of three prespecified lower-burden rule classes by patient-weighted doubly robust value, and a certificate from a paired two-stage cluster bootstrap over patients. It reports the critical exchange rate λ*, the lowest price of burden at which the rule certifies. This repository holds the code, the processed benchmark, every result table, the log of the public dataset search, and the LaTeX source of the paper.

## Reproduce

```bash
pip install -r requirements.txt   # Python 3.14
bash run_all.sh
```

`run_all.sh` downloads the three public cohorts (about 22 MB) into `out/raw`, rebuilds every table and figure, and leaves the committed outputs unchanged. The protocol takes about 5 minutes on one core and the simulations about 15 minutes on 14 cores. Compile the paper with `latexmk -pdf main.tex` in `paper/overleaf_ready`.

| File | Role | Paper |
| --- | --- | --- |
| `prep.py` | downloads the cohorts, builds one table per slice in `out/processed` | Table I, Figs 2-3 |
| `protocol.py` | the three stages, audits, and sensitivity analyses, written to `out/results` | Tables II-IV, Fig 4 |
| `stress_test.py` | data-calibrated simulations, `out/results/stress_*.csv` | Fig 5 |
| `fig1_overview.py` | the schematic overview | Fig 1 |
| `dataset_search.py` | the documented search for qualifying public cohorts, `out/search` | Section III |

Each row of `out/processed/*.csv` is one well, or one tracked organoid for Wells, with its patient, treatment family, logged arm (`low` or `high` burden), burden, state, second state channel, and endpoint. The `prep.py` docstring defines each column per slice.

## Data sources

The processed tables are derived from the following public releases and remain subject to their terms. Please cite the original studies.

- **Schuster** (pancreatic cancer, exposure duration). Schuster et al., *Nat. Commun.* 11:5271 (2020), source-data workbook, sheet "Figure 5a-b", published with the article under CC BY 4.0.
- **Spiller** (colorectal cancer, dose intensity). Spiller et al., *Front. Oncol.* 11:771173 (2021), `data/consolidated.csv` in [eitm-org/organoid_drug_response](https://github.com/eitm-org/organoid_drug_response). Copyright 2021 The University of Southern California, released for educational, research, and non-profit use.
- **Wells** (luminal breast cancer, radiation dose). Wells et al., *Sci. Rep.* 16:7928 (2026), `Figure 6/excelFilesZip.zip` in [JanesLab/WellsCJ_OrganoSeg2](https://github.com/JanesLab/WellsCJ_OrganoSeg2).

## Dataset search

`dataset_search.py` reruns the queries of the 17-18 October 2026 search and writes `queries.csv` and `records.csv`. `screening.csv` holds one row per record with the stage at which it left and the reason, and `studies.csv` groups the full-text assessments by study. Title, abstract, and full-text screening against the criteria in the `dataset_search.py` docstring was carried out with Claude (Anthropic) agents, and the qualifying cohorts were checked against their released files.

## License

The [MIT License](LICENSE) covers the code, the result tables in `out/results`, and the search log in `out/search`. The processed tables in `out/processed` keep the terms of their sources above, and the paper source in `paper/` keeps the authors' copyright, which transfers to IEEE on publication.
