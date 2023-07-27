import torch.nn as nn
import torch
import numpy as np
from torch import Tensor
from typing import Type
cos_sim = lambda a,b: np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels*self.expansion,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return  out

class ResNet(nn.Module):
    def __init__(
        self,
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int  = 1000
    ) -> None:
        super(ResNet, self).__init__()
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock`
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1

        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion

        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature
        # map should be (7, 7) for all ResNets.

        # print('Dimensions of the last convolutional feature map: ', x.shape)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def shrink_batch(inputs, targets, rp):
    label_to_ixs = {}
    for ix, label in enumerate(targets):
        label_to_ixs[label.item()] = label_to_ixs.get(label.item(), [])
        label_to_ixs[label.item()].append(ix)
    
    label_to_ixs = {label: np.array(ixs, dtype = 'int') for label, ixs in label_to_ixs.items()} 
    using_ixs = []
    for label in label_to_ixs.keys():
        threshold = rp.label_to_th[label]
#         center = inputs[label_to_ixs[label]]    
#         center = center.reshape(center.shape[0], rp.d).mean(axis = 0)
#         chosen = min([(np.linalg.norm(center - inputs[i].reshape(rp.d)), i) for i in label_to_ixs[label]])[1]
        chosen = np.random.choice(label_to_ixs[label])    
        
        using_ixs.append(chosen)
        vec = inputs[chosen].reshape(rp.d)
        using_ixs += [i for i in label_to_ixs[label] if 1-cos_sim(inputs[i].reshape(rp.d), vec)>=threshold]
    
    return using_ixs     

# Training module
def train(epoch, net, criterion, optimizer, trainloader, device, rp, p, shrinkage = 'LSH'):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    best_acc = 0
    best_loss = 999

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if shrinkage == 'LSH' and p > 0:
            ixs = shrink_batch(inputs, targets, rp)
            inputs = inputs[ixs]
            targets = targets[ixs]
        
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        ### Shrink batch
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        train_loss = train_loss / (batch_idx + 1)
        train_acc = 100. * correct / total

        if best_acc < train_acc:
            best_acc = train_acc
            best_loss = train_loss

    return best_loss, best_acc

# Testing module
def test(epoch, net, criterion, testloader, device):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    best_acc = 0
    best_loss = 999

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        test_loss = test_loss / (batch_idx + 1)
        test_acc = 100. * correct / total

        if best_acc < test_acc:
            best_acc = test_acc
            best_loss = test_loss

    return best_loss, best_acc