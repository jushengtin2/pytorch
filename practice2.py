import torch 
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    the_class = ['dog', 'cat'] #定義0是狗1是貓這樣當test出來的結果可以直接對照

    transform = transforms.Compose([
                    transforms.Resize((256, 256)), #讓照片是256*256
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#將圖像像素值標準化到均值為0.5,標準差為0.5的分佈上。
                                                                            #(0.5, 0.5, 0.5)代表RGB三個通道的均值都是0.5
                                                                            #(0.5, 0.5, 0.5)代表RGB三個通道的標準差都是0.5
    
    traindata = ImageFolder(root='pytorch/archive/dataset/training_set', transform=transform, target_transform=None)
    testdata = ImageFolder(root='pytorch/archive/dataset/test_set', transform=transform, target_transform=None)
    #ImageFolder是一個非常有用的數據集加載器,用於從文件夾結構中加載圖像數據。

    batch_size = 40

    trainloader = DataLoader(dataset= traindata, batch_size=batch_size, shuffle= True, num_workers= 2) #num_workers是讓多core cpu分工運作
    testloader = DataLoader(dataset= testdata, batch_size=batch_size, shuffle=True, num_workers=2)
    #shuffle用於決定是否在每個訓練周期（epoch）開始時隨機打亂數據。

    #為了讓多core cpu分工運作的保護措施
    #for batch_idx, (data, target) in enumerate(trainloader):
        #print('data:', data)            #純粹照片的數字資料
        #print('label:', target)         #是狗還是貓
        #break

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
        第一個捲積層conv1使用了5x5的捲積核 輸出通道為6。由於沒有提及填充 padding 和步長stride 我們假設使用默認的stride=1和padding=0。這將減少每個維度上的尺寸為4 256-5+1=252。
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

    my_model = model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(my_model.parameters(), lr=0.01, momentum=0.9) #SGD(隨機梯度下降)優化器

    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    #i.e.
    #model = Model()
    #output = model(data)
            
    """       
    參數更新方向劇烈振蕩 如果相鄰批次的梯度方向相差很大,那麼參數的更新方向也會在這兩個方向之間來回擺動,造成"鋸齒狀"的更新路徑。這種劇烈的振蕩會減緩收斂速度。
    可能陷入鞍點或梯度消失 對於一些非凸損失曲面,如果梯度方向變化過於頻繁,很可能導致參數更新無法有效向最優解收斂,反而陷入鞍點或梯度消失的區域。
    難以跳出局部最小值 如果模型陷入局部最小值區域,單純依賴當前梯度進行更新很難讓參數有足夠的動量跳出這個區域。
    """

    ############################################開始訓練###################################################

    def train_model():
        for epoch in range(2): #會遍歷兩次數據
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0): #０表示從資料的索引值０開始
                inputs, labels = data
                optimizer.zero_grad() #把累積的梯度消除因為是新的一輪了

                output = my_model(inputs)
                loss = criterion(output, labels)
                loss.backward()  #對損失值進行反向傳播，計算每個參數的梯度
                optimizer.step() #根據梯度更新模型的參數
            
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
            scheduler.step()
        print('Finished Training')

    print('Training start!')
    train_model()
        
    PATH = 'pytorch/mymodel.pth'
    torch.save(my_model.state_dict(), PATH)  #儲存我的模型
 
    ####開始測試######
    
    def imshow(img):
        img = img / 2 + 0.5      #unnormalize 會是這個公式請參考17行
        npimg = img.numpy()     #轉乘numpy數組
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  #將NumPy陣列的維度順序由(channels, height, width)變為(height, width, channels)
        plt.show()
    
    dataiter = iter(testloader)
    images, labels = next(dataiter) #next()會秀出新的一個batch的資料
    first_img = images[0]
    imshow(first_img) #把第一張圖片展出來
    #如果要秀出整個batch10張的話 可以使用torchvision.utils.make_grid(images)他會把全部都和在一起
    
    my_model = model()
    my_model.load_state_dict(torch.load(PATH))

    outputs = my_model(images)           
    probs = ff.softmax(outputs, dim=1)   #使用Softmax函數將logits轉換為概率值 （把一個向量壓縮到(0,1)範圍內的值,這些值的總和為1,可以將它們解釋為概率）
    _, preds = torch.max(probs, dim=1)   #獲取每個輸出的最大概率及其索引(索引即是預測類別) 第一個維度的max不需要因為他是很多的[] [] []去比較但根本不需要 所以只需要第二維度 （ _是一個常用的變數名,代表我們暫時不需要第一個tensor(最大值),所以被忽略掉）

    # 將預測索引映射為實際類別名稱並打印
    for i in range(len(preds)):
        print(f'{i}:  {outputs[i]} 被預測為 {the_class[preds[i]]} 答案是{the_class[labels[i]]}')