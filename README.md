# segmentation-algorithms

---
## models
### FCN U-net

---
## hame page
[github home page](https://github.com/dashboard)

---
## enviroment preparation
[enviroments](https://github.com/jhz6353/segmentation-algorithms/edit/main/requirements.txt)

---
## data preparation
we use VOC2012 to train our model
[data download](https://github.com/dataset-ninja/pascal-voc-2012/blob/main/DOWNLOAD.md)

---
## usage
### train
`CUDA_VISIBLE_DEVICES=0 python train.py --epoches 50 --batch_size 4 --datapath "dataset/VOC2012"`
