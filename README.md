# SSM's on Document AI Task
This repository contains my experiments of SSM (State Space Models) on various Document AI Task


## Current Plan:

- [ ] 1. Entity Recognition
- [ ] 2. Document Classification
- [ ] 3. Single Page Document Question Answering
- [ ] 4. Multi Page Document Question Answering


## 1. Entity Recognition

### 1.1. Dataset
* [FUNSD Dataset](https://guillaumejaume.github.io/FUNSD/)
* [CORD Dataset](https://github.com/clovaai/cord)


### 1.2. Model
For Unimodal approaches:

* [ ] 1.1. State Space Models (currently planning for [S4D](https://github.com/HazyResearch/state-spaces/blob/main/models/s4/s4d.py) model)

For Multimodal approaches:
* [ ] 1.1 LayoutLM's family model (this includes [LayoutLMv3](https://arxiv.org/abs/2204.08387))



### 1.3. Results (without pre-training, and word embeddings initialized from LayoutLMv3's word embeddings)

#### FUNSD Dataset

| Model      | Precision | Recall | F1-Score | Accuracy |
|------------|-----------|--------|----------|----------|
| S4D        | 25.30     | 39.74  | 30.91    | 54.52    |
| LayoutLMv3 | 17.59     | 32.78  | 22.89    | 42.11    |

#### CORD Dataset

| Model      | Precision | Recall | F1-Score | Accuracy |
|------------|-----------|--------|----------|----------|
| S4D        | 79.55     | 84.73  | 82.05    | 88.62    |
| LayoutLMv3 | 53.56     | 68.56  | 60.14    | 72.66    |



## 2. Document Classification

### 2.1. Dataset:
* [Tobacco 3482 Dataset](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg)

### 2.2. Model
Similar to Entity Recognition, we will be using both Unimodal and Multimodal approaches.

### 2.3. Results

| Model      | Precision | Recall | F1-Score | Accuracy |
|------------|-----------|--------|----------|----------|
| S4D        | 01.77     | 10.00  | 03.02    | 17.79    |
| LayoutLMv3 | 71.53     | 66.63  | 67.98    | 70.01    |

## 3. Single Page Document Question Answering

### 3.1. Dataset:
* [DocVQA](https://arxiv.org/abs/2007.00398)


## 4. Multi Page Document Question Answering:

### 4.1. Dataset:
* Multi Page DocVQA [data link](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=4). Paper [link](https://arxiv.org/abs/2212.05935)


## 5. References:
* Mostly all the repoistories containing the corresponding code, and my previous work (not research, but applications), google.


## 6. Citation:

```bibtex
@article{gu2022parameterization,
  title={On the parameterization and initialization of diagonal state space models},
  author={Gu, Albert and Goel, Karan and Gupta, Ankit and R{\'e}, Christopher},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={35971--35983},
  year={2022}
}
```

```bibtex
@inproceedings{huang2022layoutlmv3,
  title={Layoutlmv3: Pre-training for document ai with unified text and image masking},
  author={Huang, Yupan and Lv, Tengchao and Cui, Lei and Lu, Yutong and Wei, Furu},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={4083--4091},
  year={2022}
}
```

```bibtex
@article{tang2022unifying,
  title={Unifying Vision, Text, and Layout for Universal Document Processing},
  author={Tang, Zineng and Yang, Ziyi and Wang, Guoxin and Fang, Yuwei and Liu, Yang and Zhu, Chenguang and Zeng, Michael and Zhang, Cha and Bansal, Mohit},
  journal={arXiv preprint arXiv:2212.02623},
  year={2022}
}
```

```bibtex
@inproceedings{jaume2019,
    title = {FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
    author = {Guillaume Jaume, Hazim Kemal Ekenel, Jean-Philippe Thiran},
    booktitle = {Accepted to ICDAR-OST},
    year = {2019}
}
```

```bibtex
@article{park2019cord,
  title={CORD: A Consolidated Receipt Dataset for Post-OCR Parsing},
  author={Park, Seunghyun and Shin, Seung and Lee, Bado and Lee, Junyeop and Surh, Jaeheung and Seo, Minjoon and Lee, Hwalsuk}
  booktitle={Document Intelligence Workshop at Neural Information Processing Systems}
  year={2019}
}
```

```bibtex
@inproceedings{mathew2021docvqa,
  title={Docvqa: A dataset for vqa on document images},
  author={Mathew, Minesh and Karatzas, Dimosthenis and Jawahar, CV},
  booktitle={Proceedings of the IEEE/CVF winter conference on applications of computer vision},
  pages={2200--2209},
  year={2021}
}
```

```bibtex
@article{tito2022hierarchical,
  title={Hierarchical multimodal transformers for Multi-Page DocVQA},
  author={Tito, Rub{\`e}n and Karatzas, Dimosthenis and Valveny, Ernest},
  journal={arXiv preprint arXiv:2212.05935},
  year={2022}
}
```
