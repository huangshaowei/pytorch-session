import torch
import torchvision
import torch.nn as nn
from  torch.utils.data import DataLoader
import torch

import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 28 - 5 + 1 = 24
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 12 - 5 + 1 = 8
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
 
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # 12 * 12 * 10
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # 4 * 4 * 20= 16 * 20 = 320
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


from torchvision import datasets, transforms
trainset = datasets.MNIST(root= "./mnist", train =True, download=False,
               transform = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean=[0.5],std=[0.5])]))
testset = datasets.MNIST(root="./mnist", train=False, download=False,
               transform = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean=[0.5],std=[0.5])]))

train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)
device = torch.device("cuda")
net = MyModel().to(device)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01, momentum=0.5, weight_decay=0.0005)
sheduler = torch.optim.lr_scheduler.StepLR(optimizer,20,gamma=0.8)

def train(epoch):
    sum_loss = 0.0
    count = 0
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        pred = net(data)
        optimizer.zero_grad()
        loss = loss_func(pred, target)
        sum_loss += loss
        count+=1
        loss.backward()
        optimizer.step()
    mean_loss = sum_loss / count
    print("epoch {} loss: {}".format(epoch, mean_loss))
    return mean_loss


def test():
    net.eval()
    correct = 0.0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = net(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    print("presition: %f"%(correct / len(test_loader.dataset)))

if __name__ == "__main__":
    for epoch in range(0, 100):
        sheduler.step()
        train(epoch)
    torch.save(net, "handwrite.pth")
    net = torch.load("./handwrite.pth")
    test()

