# KBCEN

Pytorch implementation of my model.

## Environment and Dependencies
- python 3.6

In addition, please add the project folder to PYTHONPATH and `pip install` the following packages. Install the dependencies in requirements.txt or install them manually:
- torch>=1.6.0, <=1.9.0
- torchvision>=0.7.0, <=0.10.0
- numpy==1.18.5
- tqdm>=4.43.0,<4.50.0
- torchtext==0.5.0
- GitPython==3.1.0
- requests==2.23.0
- fasttext==0.9.1
- nltk==3.4.5
- editdistance==0.5.3
- transformers>=3.4.0, <=4.10.1
- sklearn==0.0
- omegaconf>=2.0.6, <=2.1
- lmdb==0.98
- termcolor==1.1.0
- iopath==0.1.8
- datasets==1.2.1
- matplotlib==3.3.4
- pycocotools==2.0.2
- ftfy==5.8
- pytorch-lightning==1.5.0
- psutil
- pillow==9.0.1

Then install the following dependencies manually:
pip install networkx torch_geometric gensim

Check that cuda versions match between your pytorch installation and torch_geometric.

## Training
To train the model on the OKVQA dataset, run the following command
```python
captionvqa_run config=captionvqa/projects/KBCEN/configs/KBCEN/okvqa/train_val.yaml run_type=train_val dataset=okvqa model=KBCEN
```

## Setup / Data
To make sure all data can be found, first run

> export captionvqa_DATA_DIR=~/.cache/torch/captionvqa/data