# SSM's on Document AI Task
This repository contains my experiments of SSM (State Space Models) on various Document AI Task


## Current Plan:

- [x] 1. Entity Recognition
- [x] 2. Document Classification
- [x] 3. Single Page Document Question Answering
- [x] 4. Multi Page Document Question Answering


## 1. Entity Recognition

### 1.1. Dataset
* [FUNSD Dataset](https://guillaumejaume.github.io/FUNSD/)
* [CORD Dataset](https://github.com/clovaai/cord)


### 1.2. Model
For Unimodal approaches:

* [x] 1.1. State Space Models (currently planning for [S4D](https://github.com/HazyResearch/state-spaces/blob/main/models/s4/s4d.py) model)
* [x] 1.2. BERT and associated models

For Multimodal approaches:
* [x] 1.1 LayoutLM's family model (this includes [LayoutLMv3](https://arxiv.org/abs/2204.08387))
* [x] 1.2 UDOP [Unifying Vision, Text, and Layout for Universal Document Processing](https://arxiv.org/abs/2212.02623)



### 1.3. Results



## 2. Document Classification

### 2.1. Dataset:
* [Tobacco 3482 Dataset](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg)

### 2.2. Model
Similar to Entity Recognition, we will be using both Unimodal and Multimodal approaches.

### 2.3. Results


## 3. Single Page Document Question Answering

### 3.1. Dataset:
* [DocVQA](https://arxiv.org/abs/2007.00398)


## 4. Multi Page Document Question Answering:

### 4.1. Dataset:
* Multi Page DocVQA [data link](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=4). Paper [link](https://arxiv.org/abs/2212.05935)