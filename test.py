import numpy as np
from torchvision import models
from dataloader import get_dataloaders
import torch
import random

# a=np.zeros((3,3))
# b=np.array([1,2,3])
# print(a)
# a[0]=b
# print(a)

# vgg16=models.vgg16(pretrained=True)
# print(vgg16)
# print(type(vgg16))
# print(vgg16.features)
#
# features=list(vgg16.features.children())
# print(features)
# print(len(features))
# print(type(features[0]))

if __name__ == '__main__':
    # root="D:/python_project/learning_project/segmentation/dataset/VOC2012"
    # print(1)
    # train_dataloader,val_dataloader=get_dataloaders(root)
    # i=0
    # print(2)
    # for img,mask in train_dataloader:
    #     print(img.shape)
    #     i+=1
    #     if i>=1:
    #         break
    myiter=[99,88,77,55]
    for i,data in enumerate(myiter):
        print(i,data)
    y=[]
    for i in range(4):
        a=[]
        for j in range(21):
            k=np.random.random((4,4))
            k=list(k)
            a.append(k)
        y.append(a)
    print(y)



    y=torch.tensor(y)
    index,y=torch.max(y,dim=1)
    print(y)
    print(y.shape)
    print(index)
    print(index.shape)
