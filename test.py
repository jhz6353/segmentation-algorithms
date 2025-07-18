import argparse
import os.path
import gc
import numpy as np
from PIL import Image
import torch
from dataloader import get_dataloaders,VOC_CLASSES,VOC_COLORMAP
from model import FCN_8s,FCN_32s
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import torchvision.transforms as transforms



def decode_segmap(segmap):
    """
    将类别索引的分割图转换为RGB彩色图像（用于可视化）
    参数:
        segmap (np.array或torch.Tensor): 形状为(H,W)的分割图，值为类别索引
    返回:
        rgb_img (np.array): 形状为(H,W,3)的RGB彩色图像
    """
    if isinstance(segmap, torch.Tensor):
        segmap = segmap.cpu().numpy()
    if len(segmap.shape) == 3:
        segmap=segmap[:,:,0]
    rgb_img = np.zeros((segmap.shape[0], segmap.shape[1], 3),dtype=np.uint8)
    for class_idx,color in enumerate(VOC_COLORMAP):
        flag= segmap==class_idx
        # print(flag)
        if flag.any():
            rgb_img[flag]=color
    return rgb_img

class test(object):
    def __init__(self,model,save_path,num_workers,gpus,num_classes):
        self.model = model
        self.save_path = save_path
        self.num_workers = num_workers
        self.gpus = gpus
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.num_classes = num_classes
        #定义损失函数和优化器
        model.to('cuda')
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    def predict_img(self,img_path,overlay):
        inputs,ori_img=self.process_img(img_path)
        inputs=inputs.to('cuda')
        self.model.eval()
        with torch.no_grad():
            pred=self.model(inputs)
            #也许有问题
            pred=torch.argmax(pred,dim=1)
            pred=pred.cpu().detach().numpy()[0]

        seg_img=decode_segmap(pred)
        if overlay:
            output=self.overlay(ori_img,seg_img)
        else:
            output=seg_img
        return output,seg_img,ori_img

    def process_img(self,img_path):
        ori_img=Image.open(img_path).convert('RGB')
        transform=transforms.Compose([
            transforms.Resize((320,320)),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        input_img=transform(ori_img)
        input_batch=input_img.unsqueeze(0)
        return input_batch,ori_img
    #叠加分割结果与原图，可视化
    def overlay(self,image, segmentation, alpha=0.65):
        image_np = np.array(image)
        segmentation_resized = np.array(Image.fromarray(segmentation.astype(np.uint8)).resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST))
        overlay = image_np.copy()
        for i in range(3):
            overlay[:, :, i] = image_np[:, :, i] * (1 - alpha) + segmentation_resized[:, :, i] * alpha
        return overlay.astype(np.uint8)


if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='FCN_8s')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--datapath', type=str, default="dataset/VOC2012")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--output_save_path', type=str, default='FCN_8s')
    parser.add_argument('--img_path', type=str, default="dataset/VOC2012")
    parser.add_argument('--checkpoint', type=str, default='FCN_8s/best_model.pth')
    parser.add_argument('--overlay', default=True, type=str2bool)
    args = parser.parse_args()

    if args.model == 'FCN_8s':
        model = FCN_8s(pretrained=True)
    elif args.model == 'FCN_32s':
        model = FCN_32s(pretrained=True)

    # _,val_dataloader=get_dataloaders(args.datapath,batch_size=args.batch_size,num_workers=args.num_workers)
    model_state_dict = torch.load(args.checkpoint, 'cuda',weights_only=False)
    model.load_state_dict(model_state_dict['model_state_dict'])
    Test=test(model,args.output_save_path,args.num_workers,args.gpus,21)
    output,seg_img,ori_img=Test.predict_img(args.img_path,args.overlay)
    plt.figure(figsize=(10,10))
    plt.subplot(1,3,1)
    plt.imshow(output)
    plt.subplot(1,3,2)
    print(seg_img.shape)
    plt.imshow(seg_img)
    plt.subplot(1,3,3)
    plt.imshow(ori_img)
    plt.show()



    '''
    CUDA_VISIBLE_DEVICES=0 python test.py --img_path "dataset/VOC2012/JPEGImages/2007_000027.jpg" --output_save_path "output"
    '''

