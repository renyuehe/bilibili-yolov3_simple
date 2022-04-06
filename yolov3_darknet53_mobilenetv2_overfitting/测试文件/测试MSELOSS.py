import torch
import torch.nn as nn


mseloss=nn.MSELoss()#均方损失函数
target = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
pred= torch.FloatTensor([[7, 8, 9], [8, 4, 3]])
cost=mseloss(pred,target)#将pred,target逐个元素求差,然后求平方,再求和,再求均值,
print(cost)#tensor(22.3333)


sum=0
for i in range (0,2):#遍历行i
    for j in range(0,3):#遍历列
        sum+=(target[i][j]-pred[i][j])*(target[i][j]-pred[i][j])#对应元素做差,然后平方
print(sum/6)#tensor(22.3333)


ret = torch.mean((target - pred) ** 2)
print(ret) #tensor(22.3333)