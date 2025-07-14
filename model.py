import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.models as models

from dataloader import NUM_CLASSES


class FCN_32s(nn.Module):
    def __init__(self,num_classes=NUM_CLASSES,pretrained=True):
        super(FCN_32s,self).__init__()
        vgg16=models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        features=list(vgg16.features.children())
        #根据FCN论文修改VGG16网络
        #前五段卷积保持不变
        self.features1=nn.Sequential(*features[0:5])
        self.features2=nn.Sequential(*features[5:10])
        self.features3=nn.Sequential(*features[10:17])
        self.features4=nn.Sequential(*features[17:24])
        self.features5=nn.Sequential(*features[24:31])
        #全连接层替换为1*1卷积
        self.features6=nn.Sequential(
            nn.Conv2d(512,4096,kernel_size=7,padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096,4096,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )
        #分类层
        self.score=nn.Conv2d(4096,num_classes,kernel_size=1)
        #上采样层
        self.upsample=nn.ConvTranspose2d(num_classes,num_classes,kernel_size=64,stride=32,padding=16,bias=False)
        #初始化参数
        self._initialize_weights()

    def forward(self,x):
        input_size=x.size()[2:]
        x=self.features1(x)
        x=self.features2(x)
        x=self.features3(x)
        x=self.features4(x)
        x=self.features5(x)
        x=self.features6(x)
        x=self.score(x)
        x=self.upsample(x)
        x=x[:,:,:input_size[0],:input_size[1]]
        return x


    def _initialize_weights(self):
        #初始化反卷积层权重为双线性上采样
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                #双线性上采样的初始化
                m.weight.data.zero_()
                m.weight.data=self._make_bilinear_weights(m.kernel_size[0],m.out_channels)

    def _make_bilinear_weights(self,size,num_channels):
        """生成双线性插值的权重"""
        facter=(size+1)//2
        if size%2==0:
            center=facter-1
        else:
            center=facter-0.5
        og=torch.FloatTensor(size,size)
        for i in range(size):
            for j in range(size):
                og[i,j]=(1-abs((i-center)/facter))*(1-abs((j-center)/facter))
        filter=torch.zeros(num_channels,num_channels,size,size)
        for i in range(num_channels):
            filter[i,i]=og
        return filter

class FCN_8s(nn.Module):
    def __init__(self,num_classes=NUM_CLASSES,pretrained=True):
        super(FCN_8s,self).__init__()
        vgg16=models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        features=list(vgg16.features.children())
        self.feature1=nn.Sequential(*features[0:5])
        self.feature2=nn.Sequential(*features[5:10])
        self.feature3=nn.Sequential(*features[10:17])
        self.feature4=nn.Sequential(*features[17:24])
        self.feature5=nn.Sequential(*features[24:31])
        self.fc6=nn.Conv2d(512,4096,kernel_size=7,padding=3)
        self.relu6=nn.ReLU(inplace=True)
        self.dropout6=nn.Dropout2d()
        self.fc7=nn.Conv2d(4096,4096,kernel_size=1)
        self.relu7=nn.ReLU(inplace=True)
        self.dropout7=nn.Dropout2d()
        #分类
        self.score_fr=nn.Conv2d(4096,num_classes,kernel_size=1)
        #pool3和pool4的1*1卷积，用于特征融合
        self.scorepool3=nn.Conv2d(256,num_classes,kernel_size=1)
        self.scorepool4=nn.Conv2d(512,num_classes,kernel_size=1)
        #两倍上采样conv7的特征
        self.upsample2_1=nn.ConvTranspose2d(num_classes,num_classes,kernel_size=4,stride=2,padding=1,bias=False)
        #2倍上采样融合后的特征
        self.upsample2_2=nn.ConvTranspose2d(num_classes,num_classes,kernel_size=4,stride=2,padding=1,bias=False)
        #8倍上采样回原始图像大小
        self.upsample8=nn.ConvTranspose2d(num_classes,num_classes,kernel_size=16,stride=8,padding=4,bias=False)
        #初始化参数
        self._initialize_weights()

    #前向传播（推理）
    def forward(self,x):
        input_size=x.size()[2:]
        x=self.feature1(x)
        x=self.feature2(x)
        pool3=self.feature3(x)
        pool4=self.feature4(pool3)
        x=self.feature5(pool4)
        x=self.fc6(x)
        x=self.relu6(x)
        x=self.dropout6(x)
        x=self.fc7(x)
        x=self.relu7(x)
        x=self.dropout7(x)
        x=self.score_fr(x)
        x=self.upsample2_1(x)
        #获取pool4的分数并裁剪
        score_pool4=self.scorepool4(pool4)
        score_pool4=score_pool4[:,:,:x.size()[2],:x.size()[3]]
        x=x+score_pool4
        #再次上采样
        x=self.upsample2_2(x)
        #获取pool分数并裁剪
        score_pool3=self.scorepool3(pool3)
        score_pool3=score_pool3[:,:,:x.size()[2],:x.size()[3]]
        x=x+score_pool3
        #8倍上采样回原始图像大小
        x=self.upsample8(x)
        #保证与输入图像大小一致
        # print('input_size:',input_size)
        x=x[:,:,:input_size[0],:input_size[1]]
        return x


    def _initialize_weights(self):
        # 初始化反卷积层权重为双线性上采样
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # 双线性上采样的初始化
                m.weight.data.zero_()
                m.weight.data = self._make_bilinear_weights(m.kernel_size[0], m.out_channels)

    def _make_bilinear_weights(self, size, num_channels):
        """生成双线性插值的权重"""
        facter = (size + 1) // 2
        if size % 2 == 0:
            center = facter - 1
        else:
            center = facter - 0.5
        og = torch.FloatTensor(size, size)
        for i in range(size):
            for j in range(size):
                og[i, j] = (1 - abs((i - center) / facter)) * (1 - abs((j - center) / facter))
        filter = torch.zeros(num_channels, num_channels, size, size)
        for i in range(num_channels):
            filter[i, i] = og
        return filter


class U_net(nn.Module):
    def __init__(self,num_classes=NUM_CLASSES,pretrained=True):
        super(U_net,self).__init__()
        #第一层 3*572*572--64*570*570--64*568*568--64*284*284
        self.conv1_1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=0)
        self.relu1_1=nn.ReLU(inplace=True)
        self.conv1_2=nn.Conv2d(64,64,kernel_size=3,stride=1,padding=0)
        self.relu1_2=nn.ReLU(inplace=True)
        self.pool1=nn.MaxPool2d(kernel_size=2)

        #第二层 64*284*284--128*282*282--128*280*280--128*140*140
        self.conv2_1=nn.Conv2d(64,128,kernel_size=3,stride=1,padding=0)
        self.relu2_1=nn.ReLU(inplace=True)
        self.conv2_2=nn.Conv2d(128,128,kernel_size=3,stride=1,padding=0)
        self.relu2_2=nn.ReLU(inplace=True)
        self.pool2=nn.MaxPool2d(kernel_size=2)

        #第三层 128*140*140--256*138*138--256*136*136--256*68*68
        self.conv3_1=nn.Conv2d(128,256,kernel_size=3,stride=1,padding=0)
        self.relu3_1=nn.ReLU(inplace=True)
        self.conv3_1=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=0)
        self.relu3_2=nn.ReLU(inplace=True)
        self.pool3=nn.MaxPool2d(kernel_size=2)

        #第四层 256*68*68--512*66*66--512*64*64--512*32*32
        self.conv4_1=nn.Conv2d(256,512,kernel_size=3,stride=1,padding=0)
        self.relu4_1=nn.ReLU(inplace=True)
        self.conv4_2=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=0)
        self.relu4_2=nn.ReLU(inplace=True)
        self.pool4=nn.MaxPool2d(kernel_size=2)

        #第五层 512*32*32--1024*30*30--1024*28*28--1024*56*56
        self.conv5_1=nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=0)
        self.relu5_1=nn.ReLU(inplace=True)
        self.conv5_2=nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=0)
        self.relu5_2=nn.ReLU(inplace=True)
        #采用转置卷积，两倍上采样
        self.upsample5=nn.ConvTranspose2d(1024,1024,kernel_size=4,stride=2,padding=1,bias=False)

        #第六层 1024*56*56--512*54*54--512*52*52--512*104*104
        self.conv6_1=nn.Conv2d(1024,512,kernel_size=3,stride=1,padding=0)
        self.relu6_1=nn.ReLU(inplace=True)
        self.conv6_2=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=0)
        self.relu6_2=nn.ReLU(inplace=True)
        self.upsample6=nn.ConvTranspose2d(512,512,kernel_size=4,stride=2,padding=1,bias=False)

        #第七层 512*104*104--256*102*102--256*100*100--256*200*200
        self.conv7_1=nn.Conv2d(512,256,kernel_size=3,stride=1,padding=0)
        self.relu7_1=nn.ReLU(inplace=True)
        self.conv7_2=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=0)
        self.relu7_2=nn.ReLU(inplace=True)
        self.upsample7=nn.ConvTranspose2d(256,256,kernel_size=4,stride=2,padding=1,bias=False)

        #第八层 256*200*200--128*198*198--128*196*196--128*392*392
        self.conv8_1=nn.Conv2d(256,128,kernel_size=3,stride=1,padding=0)
        self.relu8_1=nn.ReLU(inplace=True)
        self.conv8_2=nn.Conv2d(128,128,kernel_size=3,stride=1,padding=0)
        self.relu8_2=nn.ReLU(inplace=True)
        self.upsample8=nn.ConvTranspose2d(128,128,kernel_size=4,stride=2,padding=1,bias=False)

        #第九层（输出层）128*392*392--64*390*390--64*388*388--num_classes*388*388
        self.conv9_1=nn.Conv2d(128,64,kernel_size=3,stride=1,padding=0)
        self.relu9_1=nn.ReLU(inplace=True)
        self.conv9_2=nn.Conv2d(64,64,kernel_size=3,stride=1,padding=0)
        self.relu9_2=nn.ReLU(inplace=True)
        self.score_fr=nn.Conv2d(64,num_classes,kernel_size=1)

    def crop_tensor(self,tensor,target_size):
        delta=target_size-tensor.size()[2]
        delta=delta//2
        return tensor[:,:,delta:-delta,delta:-delta]

    def forward(self,x):
        x1=self.conv1_1(x)
        x1=self.relu1_1(x1)
        x2=self.conv1_2(x1)
        x2=self.relu1_2(x2)
        down1=self.pool1(x2)

        x3=self.conv2_1(down1)
        x3=self.relu2_1(x3)
        x4=self.conv2_2(x3)
        x4=self.relu2_2(x4)
        down2=self.pool2(x4)

        x5=self.conv3_1(down2)
        x5=self.relu3_1(x5)
        x6=self.conv3_2(x5)
        x6=self.relu3_2(x6)
        down3=self.pool3(x6)

        x7=self.conv4_1(down3)
        x7=self.relu4_1(x7)
        x8=self.conv4_2(x7)
        x8=self.relu4_2(x8)
        down4=self.pool4(x8)

        x9=self.conv5_1(down4)
        x9=self.relu5_1(x9)
        x10=self.conv5_2(x9)
        x10=self.relu5_2(x10)








