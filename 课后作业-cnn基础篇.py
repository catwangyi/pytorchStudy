import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim


class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=3,padding=1)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(20, 40,kernel_size=3,padding=1)
        self.pooling = torch.nn.MaxPool2d(kernel_size=2)
        self.fc = torch.nn.Linear(360, 10)

    def forward(self, x):
        #x---->(n, 1, 28, 28)取第0个就是batchsize
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = F.relu(self.pooling(self.conv3(x)))
        #将x(n, 20, 4, 4)转化为（n， 320）-1表示自动计算320
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


model = MyNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

batch_size = 64#每次拿64张图片
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='dataset/mnist', train=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_data = datasets.MNIST(root='dataset/mnist', train=False, transform=transform)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):

        inputs, target = data
        #GPU版本
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx %300 ==299:#每300张输出一次
            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    #不计算梯度
    with torch.no_grad():
        for data in test_loader:
            images ,labels = data
            #GPU版本
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # 每一行最大值的下标
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct +=(predicted == labels).sum().item()
    print('accuracy on test set:%f %%' % (100*correct/total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
