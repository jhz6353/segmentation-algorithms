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
### dataset
we use VOC2012 to train our model
[data download](https://github.com/dataset-ninja/pascal-voc-2012/blob/main/DOWNLOAD.md)
### recommended data file structure
--your working directory<br/>
  --dataset  
    --VOC2012  
      --Annotations  
      --ImageSets  
      --JPEGImages  
      --SegmentationClass  
      --SegmentationObject  
  --other python files  

---
## usage
### train
`python train.py --epoches 50 --batch_size 4 --datapath "dataset/VOC2012"`
### predict
`python test.py --img_path "dataset/VOC2012/JPEGImages/2007_000027.jpg" --output_save_path "output"`
