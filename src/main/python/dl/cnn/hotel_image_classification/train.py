# coding:utf-8
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
# import torch.nn.backends.sudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from dataset import ImageSet
import resnet
import pdb

model_path = './model/resnet50_places365.pth.tar'  # 保存模型的权重参数
BATCH_SIZE = 16
EPOCHES = 100
WEIGHT_DECAY = 1e-4
lr1 = 0.01
lr2 = 0.001
lr = 0.0
MOMENTUM = 0.9


def main():
    global model

    # resnet50网络结构，类别数365
    model = resnet.resnet50(num_classes=365)

    # 载入之前保存的最好模型
    checkpoint = torch.load('./model/resnet50_places365.pth.tar')

    # 加载pretrain-model，将原模型中module删去，然后load，层的名字保持一致方可加载
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    # 修改最后输出，类别数
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, 13)

    model = model.cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # Tensor image of size (C, H, W) to be normalized.
    # 训练时，图像增强
    train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                           # 仿射变换
                                           transforms.RandomAffine(5, translate=(0.1, 0.1), scale=(0.95, 1.05),
                                                                   shear=5),
                                           # Randomly change the brightness, contrast and saturation of an image.
                                           transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                                  hue=0.05),
                                           # Horizontally flip the given PIL Image randomly with a given probability.
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(), normalize])
    # 测试时，只需resize成网络输入大小224X224
    test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(), normalize])
    # 返回一个dataset对象，加载训练数据
    train_set = ImageSet('./data/train.txt', train_transforms)
    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=BATCH_SIZE, shuffle=True)
    # 加载测试数据
    test_set = ImageSet('./data/test.txt', test_transforms)
    test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=BATCH_SIZE, shuffle=False)

    # 损失函数，交叉熵损失
    criterion = nn.CrossEntropyLoss().cuda()

    # 构造SGD优化器
    # 卷积层和全连接层使用不同的学习率；
    # 卷积层参数使用place365进行初始化，学习速率不宜过大
    # fc层需要使用更大的学习速率，加快收敛
    ignored_params = list(map(id, model.fc.parameters()))
    ignored_params = ignored_params + list(map(id, model.layer4.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer = torch.optim.SGD([
        {'params': base_params},
        {'params': model.fc.parameters(), 'lr': lr1},
        {'params': model.layer4.parameters(), 'lr': lr2}
    ], lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # 训练
    for epoch in range(EPOCHES):
        # 每隔5次epoch，降低一次学习率，为上一次的0.1倍
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch)

        if (epoch + 1) % 5 == 0:
            torch.save(model, './model/train_models/net_{}.pth'.format(epoch + 1))
        test(test_loader, model)
        # 训练函数


def train(train_loader, model, criterion, optimizer, epoch):
    total = 0
    correct = 0
    # 切换为训练模式
    model.train()

    for i, (input_data, target) in enumerate(train_loader):

        # tensor to variable
        input_var = torch.autograd.Variable(input_data)
        target_var = torch.autograd.Variable(target)

        # forward 前向计算output
        input_var = input_var.cuda()
        output = model(input_var)

        # compute loss
        target_var = target_var.cuda()
        loss = criterion(output, target_var)

        # 将权重梯度初始化0, grad是累加的，每一次训练之后都要置0
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 优化器SGD更新参数
        optimizer.step()

        output = output.cpu()
        # 按行返回batch中每一行最大值的索引号[5,8,10,2,3,1...]
        _, predicted = torch.max(output.data, 1)

        # 计算batch的累积准确率，每当执行20个batchsize，输出一次结果
        total = total + target.size(0)
        correct = correct + (predicted == target).sum()

        if i % 20 == 0:
            print("===> Epoch[{}]({}/{}): Train Loss: {:.10f}".format(epoch, i, len(train_loader), loss.data[0]))
            print('Train Accuracy of the model: %f %%' % (100.0 * correct / total))
            total = 0
            correct = 0


def test(test_loader, model):
    # 切换为测试模式
    model.eval()
    total = 0
    correct = 0
    for batch in test_loader:
        input, target = torch.autograd.Variable(batch[0]), batch[1]
        input = input.cuda()
        output = model(input)
        output = output.cpu()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum()
    print('Test Accuracy of the model: %f %%' % (100.0 * correct / total))


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (0.1 ** (epoch // 5))


if __name__ == '__main__':
    main()
