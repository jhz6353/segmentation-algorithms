import os # 导入os模块，用于文件路径操作
import numpy as np # 导入numpy模块，用于数值计算
import torch # 导入PyTorch主模块
from torch.utils.data import Dataset, DataLoader # 导入数据集和数据加载器
from PIL import Image # 导入PIL图像处理库
import torchvision.transforms as transforms # 导入图像变换模块
from torchvision.transforms import functional as F # 导入函数式变换模块
import json

# VOC数据集的类别名称（21个类别，包括背景）
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa',
    'train', 'tv/monitor'
]
# 宏定义获取类别数量
NUM_CLASSES = len(VOC_CLASSES)
# 定义PIL的重采样常量
PIL_NEAREST = 0  # 最近邻重采样方式，保持锐利边缘，适用于掩码
PIL_BILINEAR = 1  # 双线性重采样方式，平滑图像，适用于原始图像
# 定义VOC数据集的颜色映射 (用于可视化分割结果)，每个类别对应一个RGB颜色
VOC_COLORMAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
]

class VocDataset(Dataset):
    def __init__(self, root,split='train',transform=None,target_transform=None,img_size=320):
        """
        初始化数据集
        参数:
            root (string): VOC数据集的根目录路径
            split (string, optional): 使用的数据集划分，可选 'train', 'val' 或 'trainval'
            transform (callable, optional): 输入图像的变换函数
            target_transform (callable, optional): 目标掩码的变换函数
            img_size (int, optional): 调整图像和掩码的大小
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.img_size = img_size
        path_idx=self.root+"/ImageSets/Segmentation/"+self.split+'.txt'
        with open(path_idx, 'r') as f:
            file_names=[x.strip() for x in f.readlines()]
        self.image_paths=[os.path.join(self.root,'JPEGImages',x+'.jpg') for x in file_names]
        self.mask_paths=[os.path.join(self.root,'SegmentationClass',x+'.png') for x in file_names]
        for image_path in self.image_paths:
            if not os.path.exists(image_path):
                print("警告: 图像文件不存在: {}".format(image_path))
            for mask_path in self.mask_paths:
                if not os.path.exists(mask_path):
                    print("警告: 掩码文件不存在: {}".format(mask_path))
        if not len(self.image_paths) == len(self.mask_paths):
            print("警告：图像和掩码数量不匹配{},{}".format(len(self.image_paths), len(self.mask_paths)))

    def __getitem__(self, index):
        """
        获取数据集中的一个样本
        参数:
            index (int): 样本索引
        返回:
            tuple: (图像, 掩码) 对，分别为图像张量和掩码张量
        """
        image=Image.open(self.image_paths[index]).convert('RGB')
        mask=Image.open(self.mask_paths[index]).convert('RGB')
        image = image.resize((self.img_size, self.img_size), PIL_BILINEAR)  # 对于图像使用双线性插值以保持平滑
        mask = mask.resize((self.img_size, self.img_size), PIL_NEAREST)  # 对于掩码使用最近邻插值以避免引入新的类别值
        if self.transform is not None:
            image = self.transform(image)
        else:
            print('警告：没有定义图像数据转换函数')
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        else:
            mask=np.array(mask)
            mask_copy=np.zeros((mask.shape[0],mask.shape[1]),dtype=np.uint8)
            if len(mask.shape) != 3 or mask.shape[2] != 3:
                raise ValueError(f"掩码维度错误: {mask.shape}, 期望为 (H,W,3)")
            for k,color in enumerate(VOC_COLORMAP):
                r=mask[:,:,0]==color[0]
                g=mask[:,:,1]==color[1]
                b=mask[:,:,2]==color[2]
                flag=r&g&b
                mask_copy[flag]=k
            mask=torch.from_numpy(mask_copy).long()
            # print('shape:',image.shape, mask.shape,index)
        return image,mask

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.image_paths)

    def set_split(self, split):
        """设置数据集的类别（训练、验证、测试）"""
        self.split=split

def get_transforms(train=True):
    """
     获取图像变换函数
     参数:
         train (bool): 是否为训练集，决定是否应用数据增强
     返回:
         tuple: (图像变换, 目标掩码变换)
     """
    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    target_transform = None #掩码已在_getitem_中处理
    return transform,target_transform


def get_dataloaders(voc_root, batch_size=4, num_workers=4, img_size=320):
    """
    创建训练和验证数据加载器
    参数:
        voc_root (string): VOC数据集根目录
        batch_size (int): 批次大小
        num_workers (int): 数据加载的线程数
        img_size (int): 图像的大小
    返回:
        tuple: (train_loader, val_loader) 训练和验证数据加载器
    """
    transform_train,target_transform_train=get_transforms(train=True)
    transform_val,target_transform_val=get_transforms(train=False)
    train_dataset=VocDataset(root=voc_root,split='train',transform=transform_train,target_transform=target_transform_train,img_size=img_size)
    val_dataset=VocDataset(root=voc_root,split='val',transform=transform_val,target_transform=target_transform_val,img_size=img_size)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return train_dataloader, val_dataloader



