# LUQ: Long-text Uncertainty Quantification for LLMs

This repo is for the paper [LUQ: Long-text Uncertainty Quantification for LLMs](https://arxiv.org/abs/2403.20279).

**Update:** We provide a more comprehensive repo for *more UQ methods and more datasets on long-form generation*: https://github.com/caiqizh/atomic_calibration, which is for the followup work: [Atomic Calibration of LLMs in Long-Form Generations](https://arxiv.org/abs/2410.13246).

**Update:** We have recently included the more advanced Llama3-8b-instruct as our NLI tool. By utilizing VLLM, we can significantly increase the speed of inference and achieve better performance. 

```
@inproceedings{zhang-etal-2024-luq,
    title = "{LUQ}: Long-text Uncertainty Quantification for {LLM}s",
    author = "Zhang, Caiqi  and
      Liu, Fangyu  and
      Basaldella, Marco  and
      Collier, Nigel",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.299/",
    doi = "10.18653/v1/2024.emnlp-main.299",
    pages = "5244--5262",
}

@misc{zhang2024atomic,
    title={Atomic Calibration of LLMs in Long-Form Generations},
    author={Caiqi Zhang and Ruihan Yang and Zhisong Zhang and Xinting Huang and Sen Yang and Dong Yu and Nigel Collier},
    year={2024},
    eprint={2410.13246},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
