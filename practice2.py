import torch 
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import multiprocessing

transform = transforms.Compose([
                  transforms.Resize((256, 256)), #讓照片是256*256
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
traindata = ImageFolder('pytorch/archive/dataset/training_set', transform=transform, target_transform=None)
testdata = ImageFolder('pytorch/archive/dataset/test_set', transform=transform, target_transform=None)
#ImageFolder是一個非常有用的數據集加載器,用於從文件夾結構中加載圖像數據。

batch_size = 1

trainloader = DataLoader(dataset=traindata, batch_size=batch_size, shuffle= True, num_workers= 2) #num_workers是讓多core cpu分工運作
testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=2)
#shuffle用於決定是否在每個訓練周期（epoch）開始時隨機打亂數據。


if __name__ == '__main__':   #為了讓多core cpu分工運作的保護措施
    for batch_idx, (data, target) in enumerate(trainloader):
        print('data:', data)            #純粹照片的數字資料
        print('label:', target)         #是狗還是貓
        break

import torch.nn as nn
import torch.nn.functional as ff

class model(nn.Module):  #使用模型輸入資料進行預測時，就會直接執行 forward()，不需再另外調用。
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 6, 5) #輸入通道數量是3因為我們的資料是RGB 輸出是6 捲積核5*5
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 61 * 61, 120)  #fc是全連接層
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)      #輸出通道數量是2因為我們的資料是貓狗
    """
    第一個捲積層conv1使用了5x5的捲積核 輸出通道為6。由於沒有提及填充 padding 和步長stride 我們假設使用默認的stride=1和padding=0。這將減少每個維度上的尺寸為4（256-5+1=252）。
    緊接著是一個池化層 視窗大小為2x2 stride=2。這將把特徵圖的尺寸減半252/2=126。
    第二個捲積層conv2同樣使用了5x5的捲積核 輸出通道為16 同樣假設stride=1和padding=0。這會再次減少尺寸為4 126-5+1=122 。
    另一個相同配置的池化層會把尺寸減半 122/2=61 。
    基於上述計算 最後一次池化層輸出的特徵圖大小為61x61 且有16個這樣的特徵圖 對應於conv2的輸出通道數 。
    """


    def forward(self, x):   #定義前向傳播函數
        x = self.pool(ff.relu(self.conv1(x)))
        x = self.pool(ff.relu(self.conv2(x)))
        x = torch.flatten(x, 1) #將捲積層輸出的多維特徵圖轉換成一維向量，以便全連接層可以處理。
        x = ff.relu(self.fc1(x))
        x = ff.relu(self.fc2(x))
        x = self.fc3(x)
        return x



#i.e.
#model = Model()
#output = model(data)
        
"""       
my_loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) #SGD(隨機梯度下降)優化器

參數更新方向劇烈振蕩 如果相鄰批次的梯度方向相差很大,那麼參數的更新方向也會在這兩個方向之間來回擺動,造成"鋸齒狀"的更新路徑。這種劇烈的振蕩會減緩收斂速度。
可能陷入鞍點或梯度消失 對於一些非凸損失曲面,如果梯度方向變化過於頻繁,很可能導致參數更新無法有效向最優解收斂,反而陷入鞍點或梯度消失的區域。
難以跳出局部最小值 如果模型陷入局部最小值區域,單純依賴當前梯度進行更新很難讓參數有足夠的動量跳出這個區域。


#開始訓練#
epoch_size = 30
for epoch in range(epoch_size):
    for batch_idx, (data, target) in enumerate(trainloader):
"""
