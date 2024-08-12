# HD-LoA Prompting

This repository contains the code for our [paper:](https://arxiv.org/abs/2311.06555)

**"LLMs Learn Task Heuristics from Demonstrations: A Heuristic-Driven Prompting Strategy for Document-Level Event Argument Extraction" (ACL 2024)**

---
## Data
DocEE dataset can be download from [this GitHub repository](https://github.com/tongmeihan1995/DocEE).

Please ensure data directory follows the structure below:

```plaintext
HD_LoA/
├── RAMS/
│   └── RAMS_1_0/
│       └── data/
│           ├── train.jsonlines
│           ├── dev.jsonlines
│           └── test.jsonlines
└── DocEE/
    └── data/
        ├── normal_setting/
        │   ├── train.json
        │   ├── dev.json
        │   └── test.json
        └── cross_domain_setting/
            ├── train_source_domain.json
            ├── train_targe_domain.json
            └── dev_targe_domain.json
            └── test_targe_domain.json
```

## Run Experiment

To run HD-LoA prompting using the following command:

```bash
python main.py --dataset_name <dataset> --data_type <type> --model_type <model>
```

## Acknowledgment

The code for accuracy evaluation is from [PAIE](https://github.com/mayubo2333/PAIE). We appreciate their excellent contributions!

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{hdloaprompting2024,
  title={LLMs Learn Task Heuristics from Demonstrations: A Heuristic-Driven Prompting Strategy for Document-Level Event Argument Extraction},
  author={Zhou, Hanzhang  and
    Qian, Junlang and
    Feng, Zijian and
    Lu, Hui and
    Zhu, Zixiao  and
    Mao, Kezhi},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)"},
  year={2024}
}
```
