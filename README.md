# LMGC
This repository contains the source code for our paper [A Language Model-based Generative Classifier for Sentence-level Discourse Parsing](https://aclanthology.org/2021.emnlp-main.188/).

## Getting Started
### Requirements


### Clone this repository 
```sh
git clone https://github.com/zhangying9128/LMGC.git
```

### Install fairseq from our repository
Please use our modified fairseq.
```sh
cd mpnet_by_microsoft/fairseq/
pip install --editable .
cd ../..
```

### Download pre-trained MPNet
Before reproducing our work, please download pre-trained MPNet-base model.
```sh
wget https://modelrelease.blob.core.windows.net/pre-training/MPNet/mpnet.base.tar.gz
tar -xf mpnet.base.tar.gz
```

### Our trained models
As we mentioned in our paper, we run 5 trials with random seeds. You can use the following LMGC models to reproduce our results.
Please save our trained models to the corresponding folders in [checkpoints](https://github.com/zhangying9128/LMGC/tree/main/checkpoints).
| Model |Trial| Link|
|---|---|---|
| Enhance_e| 1 | [download (.pt)]() | 
| Enhance_e| 2 | [download (.pt)]() | 
| Enhance_e| 3 | [download (.pt)]() | 
| Enhance_e| 4 | [download (.pt)]() | 
| Enhance_e| 5 | [download (.pt)]() | 
| Extend_e| 1 | [download (.pt)]() | 
| Extend_e| 2 | [download (.pt)]() | 
| Extend_e| 3 | [download (.pt)]() | 
| Extend_e| 4 | [download (.pt)]() | 
| Extend_e| 5 | [download (.pt)]() | 
|---|---|---|
| Enhance_r| 1 | [download (.pt)]() | 
| Enhance_r| 2 | [download (.pt)]() | 
| Enhance_r| 3 | [download (.pt)]() | 
| Enhance_r| 4 | [download (.pt)]() | 
| Enhance_r| 5 | [download (.pt)]() | 

### Data Preprocessing
Please download [RST Discourse Treebank](https://catalog.ldc.upenn.edu/LDC2002T07) dataset and prepare candidate files by yourself. 
To utilize our code, you need to use our corresponding data format listed in [data](https://github.com/zhangying9128/LMGC/tree/main/data) folder. If you would to reproduce our results, please prove that you have the License for downloading RST Discourse Treebank, so that we could provide our pre-processed data with candidate sentences to you.
And then use our following script to binary data. Please edit `MAIN_TASK` and `SUBTASK` based on your setting.
```sh
bash commands/preprocess.sh
```

### Training
You can train LMGC with the following scripts on a GPU.
```sh
bash commands/train.sh
```


### Evaluation
You can use our trained LMGC for reranking with the following script.
Please edit `MAIN_TASK`, `SUBTASK`, `LABEL_EMBEDDING`, `DATA_DIR`, `MODEL_DIR`, and `REFERENCE_DIR` based on your setting.
```sh
bash commands/score.sh
```

## Citation:
Please cite as:
```bibtex
@inproceedings{zhang-etal-2021-language,
    title = "A Language Model-based Generative Classifier for Sentence-level Discourse Parsing",
    author = "Zhang, Ying  and
      Kamigaito, Hidetaka  and
      Okumura, Manabu",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.188",
    doi = "10.18653/v1/2021.emnlp-main.188",
    pages = "2432--2446",
    abstract = "Discourse segmentation and sentence-level discourse parsing play important roles for various NLP tasks to consider textual coherence. Despite recent achievements in both tasks, there is still room for improvement due to the scarcity of labeled data. To solve the problem, we propose a language model-based generative classifier (LMGC) for using more information from labels by treating the labels as an input while enhancing label representations by embedding descriptions for each label. Moreover, since this enables LMGC to make ready the representations for labels, unseen in the pre-training step, we can effectively use a pre-trained language model in LMGC. Experimental results on the RST-DT dataset show that our LMGC achieved the state-of-the-art F1 score of 96.72 in discourse segmentation. It further achieved the state-of-the-art relation F1 scores of 84.69 with gold EDU boundaries and 81.18 with automatically segmented boundaries, respectively, in sentence-level discourse parsing.",
}
```