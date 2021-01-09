import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
from PIL import Image
from torch.utils.data import Dataset
import time
import datetime

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 30   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 64   #批处理尺寸(batch_size)
# LR = 0.0001        #学习率

class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform = None):
        fh = open(txt_path, 'r', encoding='utf-8')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.imgs)

class ResNet(nn.Module):
    def __init__(self, num_classes=685):   # num_classes，此处为 二分类值为2
        super(ResNet, self).__init__()
        net = models.resnet18(pretrained=True)   # 从预训练模型加载VGG16网络参数
        net.classifier = nn.Sequential()  # 将分类层置空，下面将改变我们的分类层
        self.features = net  # 保留VGG16的特征层
        self.classifier = nn.Sequential(    # 定义自己的分类层
                nn.Linear(1000, 1000),  #512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
                nn.ReLU(True),
                nn.Dropout(0.5),
#                 nn.Linear(1024, 1024),
#                 nn.ReLU(True),
#                 nn.Dropout(0.3),
                nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

net = ResNet().to(device)

from collections.abc import Iterable
def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze

def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.Resize((150, 150)),

    transforms.RandomHorizontalFlip(0.5),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.RandomVerticalFlip(0.5),  # 竖直翻转
    transforms.RandomRotation(30),
    transforms.RandomCrop(128, padding=4),
    #     transforms.ColorJitter(brightness=0.5),
    #     transforms.ColorJitter(contrast=0),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图像大小
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

train_datasets = MyDataset(r'./train_path_0.txt', transform=transform_train)
test_datasets = MyDataset(r'./test_path_0.txt', transform=transform_test)
trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(test_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


if __name__ == '__main__':
    print("Start Training!")  # 定义遍历数据集的次数
    start_time = time.time()

    criterion = nn.CrossEntropyLoss()
    criterion.to(device=device)

    for epoch in range(pre_epoch, EPOCH):
        if epoch == 0:
            optimizer = optim.Adam(net.parameters(), lr=0.0001)
            # 冻结 fc1, fc3层
            freeze_by_names(net, ('features'))
        elif epoch == 11:
            torch.save(net, r'./model/temp_model.pth')
            optimizer = optim.Adam(net.parameters(), lr=0.00005)
            # 解冻em, fc1, fc3层
            unfreeze_by_names(net, ('features'))
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            # 准备数据
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()

            maxk = max((1, 5))
            label_resize = labels.view(-1, 1)
            _, predicted = outputs.topk(maxk, 1, True, True)
            total += labels.size(0)
            correct += torch.eq(predicted, label_resize).cpu().sum().float().item()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

        # 每训练完一个epoch测试一下准确率
        print("Waiting Test!")
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)

                maxk = max((1, 5))
                label_resize = labels.view(-1, 1)
                _, predicted = outputs.topk(maxk, 1, True, True)
                total += labels.size(0)
                correct += torch.eq(predicted, label_resize).cpu().sum().float().item()

            print('测试分类准确率为：%.3f%%' % (100 * correct / total))
            acc = 100. * correct / total

    print("Training Finished, TotalEPOCH=%d" % EPOCH)
    end_time = time.time()
    print(str(datetime.timedelta(seconds=(end_time-start_time))))
