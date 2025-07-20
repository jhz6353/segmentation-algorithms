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
        if flag.any():
            rgb_img[flag]=color
    return rgb_img

class Trainer(object):
    def __init__(self,train_dataloader,val_dataloader,model,epoches,lr,save_path,batch_size,num_workers,gpus,resume):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.epoches=epoches
        self.save_path = save_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.gpus = gpus
        self.lr = lr
        self.resume=resume
        self.momentum = 0.9
        self.weight_decay = 1e-4
        #定义损失函数和优化器
        self.criterion=torch.nn.CrossEntropyLoss(ignore_index=255)
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.lr,momentum=self.momentum, weight_decay=self.weight_decay)
        self.schedule=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=10,gamma=0.1)
        model.to('cuda')

    def train(self):
        print('训练样本数：{},验证样本数：{}'.format(len(self.train_dataloader.dataset),len(self.val_dataloader.dataset)))
        best_miou=0.0
        start_epoch=0
        if self.resume is not None:
            if os.path.isfile(self.resume):
                checkpoint = torch.load(self.resume,weights_only=False)
                start_epoch = checkpoint['epoch']
                best_miou=checkpoint['best_miou']
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.schedule.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                print('从{}恢复训练'.format(start_epoch))
            else:
                raise FileNotFoundError

        history={
            "train_loss":[],
            'val_loss':[],
            'miou':[],
            'pixel_accuracy':[]
        }

        for epoch in range(start_epoch,self.epoches):
            t0=time.time()
            train_loss=0.0
            batch_count=0
            self.model.train()
            print(epoch)
            for img,target in tqdm(self.train_dataloader):
                img=img.to('cuda')
                target=target.to('cuda')
                self.optimizer.zero_grad()
                y_pred=self.model(img)
                loss=self.criterion(y_pred,target)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()*img.size(0)
                batch_count+=1

                if batch_count % 10 == 0:
                    torch.cuda.empty_cache()

                del img,target,y_pred,loss

            train_loss=train_loss/self.train_dataloader.dataset.__len__()
            history['train_loss'].append(train_loss)
            #更新学习率，清空显存
            self.schedule.step()
            gc.collect()
            torch.cuda.empty_cache()
            #计算验证集损失和准确率等
            val_loss,miou,iou_cls,pixel_accuracy=self.val()
            history['val_loss'].append(val_loss)
            history['miou'].append(miou)
            history['pixel_accuracy'].append(pixel_accuracy)
            epoch_time=time.time()-t0
            #打印相关参数
            print('epoch{}time:{},train_loss:{},val_loss:{},miou:{},pixel_accuracy:{}'.format(epoch,epoch_time,train_loss,val_loss,miou,pixel_accuracy))
            #保存最佳模型和最新模型
            if miou>best_miou:
                best_miou=miou
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'lr_scheduler_state_dict': self.schedule.state_dict(),
                        'best_miou': best_miou
                    },
                    os.path.join(self.save_path,'best_model.pth')
                )

            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'lr_scheduler_state_dict': self.schedule.state_dict(),
                    'best_miou': best_miou
                },
                os.path.join(self.save_path,'last_model.pth')
            )
            #可视化

        #绘制曲线
        self.draw(history)

    def val(self):
        total_corrects=0.0
        total_pixels=0
        self.model.eval()
        val_loss=0.0
        miou=0.0
        iou_cls=np.zeros(21)
        num_cls=np.zeros(21)
        with torch.no_grad():
            for img,target in tqdm(self.val_dataloader):
                img=img.to('cuda')
                target=target.to('cuda')
                y_pred=self.model(img)
                #计算损失函数
                loss=self.criterion(y_pred,target)
                val_loss += loss.item()*img.size(0)
                #把所有通道合并
                _,pred=torch.max(y_pred,1)

                #计算像素准确率
                correct= (pred==target).sum().item()
                total_corrects+=correct
                total_pixels+=target.numel()

                #计算每一类的总iou
                for cls in range(len(VOC_CLASSES)):
                    pred_cls=pred==cls
                    target_cls=target==cls
                    intersection=(pred_cls&target_cls).sum().item()
                    union=(pred_cls|target_cls).sum().item()
                    if union>0:
                        iou_cls[cls]+=intersection/union
                        num_cls[cls]+=1
                del img,target,y_pred,loss,pred
                torch.cuda.empty_cache()


        #计算平均准确率和每一类的平均iou
        mean_corrects=total_corrects/total_pixels
        for cls in range(len(VOC_CLASSES)):
            if num_cls[cls]>0:
                iou_cls[cls]/=num_cls[cls]

        #计算miou
        miou=np.mean(iou_cls)
        #计算验证集的平均损失
        val_loss=val_loss/self.val_dataloader.dataset.__len__()
        #释放显存
        gc.collect()
        torch.cuda.empty_cache()

        return val_loss,miou,iou_cls,mean_corrects

    def draw(self,history):
        plt.subplot(2,2,1)
        plt.plot(range(self.epoches),history['train_loss'])
        plt.xlabel('epoch')
        plt.ylabel('train_loss')
        plt.title('train_loss')

        plt.subplot(2,2,2)
        plt.plot(range(self.epoches),history['val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('val_loss')
        plt.title('val_loss')

        plt.subplot(2,2,3)
        plt.plot(range(self.epoches),history['miou'])
        plt.xlabel('epoch')
        plt.ylabel('miou')
        plt.title('miou')

        plt.subplot(2,2,4)
        plt.plot(range(self.epoches),history['pixel_accuracy'])
        plt.xlabel('epoch')
        plt.ylabel('pixel_accuracy')
        plt.title('pixel_accuracy')

        plt.show()

    def visualize(self):
        return 0

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
    parser.add_argument('--epoches', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='FCN_8s')
    parser.add_argument('--datapath', type=str, default="dataset/VOC2012")
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    train_dataloader,val_dataloader=get_dataloaders(args.datapath,batch_size=args.batch_size,num_workers=args.num_workers)
    print('successfully build dataloaders')
    if args.model == 'FCN_8s':
        model = FCN_8s(pretrained=True)
    elif args.model == 'FCN_32s':
        model = FCN_32s(pretrained=True)

    trainer=Trainer(train_dataloader,val_dataloader,model,args.epoches,args.lr,args.save_path,args.batch_size,args.num_workers,args.gpus,args.resume)
    trainer.train()

    '''
    CUDA_VISIBLE_DEVICES=0 python train.py --epoches 50 --batch_size 4 --datapath "dataset/VOC2012"

    '''