import torch 
import numpy as np

a = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(a)

b = torch.tensor([[1, 2],[3, 4],[5, 6]], dtype=torch.float64)
print(b)

c = torch.zeros([2, 2])
d = torch.ones([3, 3])
print(c, d)

print(torch.cuda.is_available()) #看是否能用gpu
print(torch.cuda.device_count()) #能用幾個


torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float64, requires_grad=True)
#requires_grad=True讓 PyTorch 可以使用此資料來自動跟蹤整個計算圖

# 建立隨機數值的 Tensor 並設定 requires_grad=True
x = torch.randn(2, 3, requires_grad=True)
y = torch.randn(2, 3, requires_grad=True)
z = torch.randn(2, 3, requires_grad=True)
# 計算式子
a = x * y
b = a + z
c = torch.sum(b)
# 計算梯度
c.backward()
# 查看 x 的梯度值
print(x.grad)

