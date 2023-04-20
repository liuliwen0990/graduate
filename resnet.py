from torch import nn
import torch as t
from torch.nn import functional as F
from torch.autograd import Variable as V



from audioop import bias
import torch
from torch.nn import init
import torchvision.transforms as transforms
import cv2
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import pandas as pd
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics
import xlwt
import matplotlib.pyplot as plt
import math
from pytorchtools import EarlyStopping
import time
import random
from PIL import Image
import os
from torch.utils.data import Dataset
transforms = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

class Mydata_sets(Dataset):
    def __init__(self, txt, transform=None, target_transform=None):
        super(Mydata_sets, self).__init__()
        self.txt = txt
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            #print(words[1])
            label_raw = words[1].split(',')
            label = [int(x) for x in label_raw]
            label = torch.Tensor(label)
            #imgs.append((words[0], list(map(int,words[1,:]))))  # imgs中包含有图像路径和标签
            imgs.append((words[0], label))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        label = label.to(device)
        img = Image.open(os.path.join(os.path.dirname(self.txt), fn))
        #img = img.crop((14, 14, 128-14, 128-14))
        #img = img.resize((128, 128), Image.LANCZOS)
        #print(img.size)
        #img = np.array(img).astype("float32").transpose((2, 0, 1)) / 255
        if self.transform is not None:
            img = self.transform(img)
            img = img.to(device)
        return img, label

    def __len__(self):
        return len(self.imgs)
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5 * math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def compute_rank(y_prob):
    rank = np.zeros(y_prob.shape)
    for i in range(len(y_prob)):
        temp = y_prob[i, :].argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(y_prob[i, :]))
        rank[i, :] = ranks
    return y_prob.shape[1] - rank


def compute_ranking_loss(y_prob, label):
    # y_predict = y_prob > 0.5
    num_samples, num_labels = label.shape
    loss = 0
    for i in range(num_samples):
        prob_positive = y_prob[i, label[i, :] > 0.5]
        prob_negative = y_prob[i, label[i, :] < 0.5]
        s = 0
        for j in range(prob_positive.shape[0]):
            for k in range(prob_negative.shape[0]):
                if prob_negative[k] >= prob_positive[j]:
                    s += 1

        label_positive = np.sum(label[i, :] > 0.5)
        label_negative = np.sum(label[i, :] < 0.5)
        if label_negative != 0 and label_positive != 0:
            loss = loss + s * 1.0 / (label_negative * label_positive)

    return loss * 1.0 / num_samples


def compute_one_error(y_prob, label):
    num_samples, num_labels = label.shape
    loss = 0
    for i in range(num_samples):
        pos = np.argmax(y_prob[i, :])
        loss += label[i, pos] < 0.5
    return loss * 1.0 / num_samples


def compute_coverage(y_prob, label):
    num_samples, num_labels = label.shape
    rank = compute_rank(y_prob)
    coverage = 0
    for i in range(num_samples):
        if sum(label[i, :] > 0.5) > 0:
            coverage += max(rank[i, label[i, :] > 0.5])
    coverage = coverage * 1.0 / num_samples - 1
    return coverage / num_labels


def compute_average_precision(y_prob, label):
    num_samples, num_labels = label.shape
    rank = compute_rank(y_prob)
    precision = 0
    for i in range(num_samples):
        positive = np.sum(label[i, :] > 0.5)
        rank_i = rank[i, label[i, :] > 0.5]
        temp = rank_i.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(rank_i))
        ranks = ranks + 1
        ans = ranks * 1.0 / rank_i
        if positive > 0:
            precision += np.sum(ans) * 1.0 / positive
    return precision / num_samples


def compute_macro_auc(y_prob, label):
    n, m = label.shape
    macro_auc = 0
    valid_labels = 0
    for i in range(m):
        if np.unique(label[:, i]).shape[0] == 2:
            index = np.argsort(y_prob[:, i])
            pred = y_prob[:, i][index]
            y = label[:, i][index] + 1
            fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
            temp = metrics.auc(fpr, tpr)
            macro_auc += temp
            valid_labels += 1
    macro_auc /= valid_labels
    return macro_auc


def label_transform(label):
    num_example, num_label = label.shape
    label = label.tolist()
    label_new = np.zeros((num_example,), dtype=np.int)
    # print(label)
    for i in range(0, num_example):
        label_new[i] = label[i].index(1.0)
    # print(label_new)
    return label_new


def label_predict(output,threshold = 0.5):
    num_examples,num_labels = output.shape
    predict_label = np.zeros((num_examples,num_labels))
    for i in range(0,num_examples):
        for j in range(0,num_labels):
            if output[i,j]>=threshold:
                predict_label[i,j]=1
        if not 1 in predict_label[i,:]:
            index = np.where(output[i, :] == max(output[i, :]))
            predict_label[i, index] = 1

    return predict_label
def false_labels(true_label, predict_label):
    num_examples, num_labels = predict_label.shape
    false_label = 0
    for i in range(0, num_examples):
        for j in range(0, num_labels):
            if true_label[i, j] - predict_label[i, j] != 0:
                false_label = false_label + 1
    # hamming_loss = false_labels/(num_labels*num_examples)
    return false_label


def compute_macro_f1(pred_label, label):
    up = np.sum(pred_label * label, axis=0)
    down = np.sum(pred_label, axis=0) + np.sum(label, axis=0)
    if np.sum(np.sum(label, axis=0) == 0) > 0:
        up[down == 0] = 0
        down[down == 0] = 1
    macro_f1 = 2.0 * np.sum(up / down)
    macro_f1 = macro_f1 * 1.0 / label.shape[1]
    return macro_f1


class ResidualBlock(nn.Module): # 定义ResidualBlock类 （11）
    """实现子modual：residualblock"""
    def __init__(self,inchannel,outchannel,stride=1,shortcut=None): # 初始化，自动执行 （12）
        super(ResidualBlock, self).__init__() # 继承nn.Module （13）
        self.left = nn.Sequential(  # 左网络，构建Sequential，属于特殊的module，类似于forward前向传播函数，同样的方式调用执行 （14）（31）
            nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut # 右网络，也属于Sequential，见（8）步可知，并且充当残差和非残差的判断标志。 （15）
        
    def forward(self,x): # ResidualBlock的前向传播函数 （29）
        out = self.left(x) # # 和调用forward一样如此调用left这个Sequential（30）
        if self.right is None: # 残差（ResidualBlock）（32）
            residual = x  #（33）
        else: # 非残差（非ResidualBlock） （34）
            residual = self.right(x) # （35）
        out += residual # 结果相加 （36）
        #print(out.size()) # 检查每单元的输出的通道数 （37）
        return F.relu(out) # 返回激活函数执行后的结果作为下个单元的输入 （38）

class ResNet(nn.Module): # 定义ResNet类，也就是构建残差网络结构 （2）
    """实现主module：ResNet34"""
    def __init__(self,numclasses=1000): # 创建实例时直接初始化 （3）
        super(ResNet, self).__init__() # 表示ResNet继承nn.Module （4）
        self.pre = nn.Sequential( # 构建Sequential，属于特殊的module，类似于forward前向传播函数，同样的方式调用执行 （5）（26）
            nn.Conv2d(3,64,7,2,3,bias=False),  # 卷积层，输入通道数为3，输出通道数为64，包含在Sequential的子module，层层按顺序自动执行
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(3,2,1)
        )

        self.layer1 = self.make_layer(64,128,4) # 输入通道数为64，输出为128，根据残差网络结构将一个非Residual Block加上多个Residual Block构造成一层layer（6）
        self.layer2 = self.make_layer(128,256,4,stride=2) #  输入通道数为128，输出为256 （18，流程重复所以标注省略7-17过程）
        self.layer3 = self.make_layer(256,256,6,stride=2) #  输入通道数为256，输出为256 （19，流程重复所以标注省略7-17过程）
        self.layer4 = self.make_layer(256,512,3,stride=2) #  输入通道数为256，输出为512 （20，流程重复所以标注省略7-17过程）

        self.fc = nn.Linear(512,5) # 全连接层，属于残差网络结构的最后一层，输入通道数为512，输出为numclasses （21）

    def make_layer(self,inchannel,outchannel,block_num,stride=1): # 创建layer层，（block_num-1）表示此层中Residual Block的个数 （7）
        """构建layer，包含多个residualblock"""
        shortcut = nn.Sequential( # 构建Sequential，属于特殊的module，类似于forward前向传播函数，同样的方式调用执行 （8）
            nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = [] # 创建一个列表，将非Residual Block和多个Residual Block装进去 （9）
        layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut)) # 非残差也就是非Residual Block创建及入列表 （10）

        for i in range(1,block_num):
            layers.append(ResidualBlock(outchannel,outchannel)) # 残差也就是Residual Block创建及入列表 （16）

        return nn.Sequential(*layers) # 通过nn.Sequential函数将列表通过非关键字参数的形式传入，并构成一个新的网络结构以Sequential形式构成，一个非Residual Block和多个Residual Block分别成为此Sequential的子module，层层按顺序自动执行，并且类似于forward前向传播函数，同样的方式调用执行 （17） （28）

    def forward(self,x): # ResNet类的前向传播函数 （24）
        x = self.pre(x)  # 和调用forward一样如此调用pre这个Sequential（25）

        x = self.layer1(x) # 和调用forward一样如此调用layer1这个Sequential（27）
        x = self.layer2(x) # 和调用forward一样如此调用layer2这个Sequential（39，流程重复所以标注省略28-38过程）
        x = self.layer3(x) # 和调用forward一样如此调用layer3这个Sequential（40，流程重复所以标注省略28-38过程）
        x = self.layer4(x) # 和调用forward一样如此调用layer4这个Sequential（41，流程重复所以标注省略28-38过程）

        x = F.avg_pool2d(x,7) # 平均池化 （42）
        x = x.view(x.size(0),-1) # 设置返回结果的尺度 （43）
        return self.fc(x) # 返回结果 （44）


#train_snr_val = ['-20','-15','-10','-5','0','5','10','15','20']
train_snr_val = ['-20','-15','-10','-5','0','5','10','15','20']
test_snr_val = ['-20','-15','-10','-5','0','5','10','15','20']
BATCH_SIZE = 8
num_labels = 5
count = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for snr in train_snr_val:
    root1 = r"image_train/images_snr_{}.txt".format(snr)
    if count == 0:
        my_train_datasets = Mydata_sets(root1, transform=transforms)
    else:
        my_train_datasets += Mydata_sets(root1, transform=transforms)
    count += 1

count = 0
for snr in test_snr_val:
    root2 = r"image_test/images_snr_{}.txt".format(snr)
    if count == 0:
        my_test_datasets = Mydata_sets(root2, transform=transforms)
    else:
        my_test_datasets += Mydata_sets(root2, transform=transforms)
    count += 1



train_size = int(1 * len(my_train_datasets))
valid_size = len(my_train_datasets)-train_size
#my_train_datasets = my_train_datasets.cuda()


train_dataset, valid_dataset = torch.utils.data.random_split(my_train_datasets, [train_size, valid_size])
train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True)

# 这里将valid数据集合改成了验证集
valid_loader = Data.DataLoader(
            dataset=my_test_datasets,
            shuffle=True)


model = ResNet() # 创建ResNet残差网络结构的模型的实例  (1)

model.apply(weight_init)
model = model.to(device)
m = nn.Sigmoid()
loss_func = nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)  # 论文就是0.01
n_epoch = 30
patience = 30  # 当验证集损失在连续15次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
early_stopping = EarlyStopping(patience, verbose=True)  # 关于 EarlyStopping 的代码可先看博客后面的内容
macro_auc_count = []
hamming_loss_count = []
#train_losses = []
valid_losses = []
valid_hammingloss = []
#avg_train_losses = []
#avg_valid_losses = []
#train_losses = []
valid_losses = []
start = time.time()
for epoch in range(1, n_epoch + 1):

    print('epoch=', epoch)
    model.train()
    for i, (x, y) in enumerate(train_loader):  # i:batch,x:data;y:target
        batch_x = Variable(x)  # torch.Size([128, 1, 28, 28])
        batch_y = Variable(y)  # torch.Size([128])

        # 获取最后输出
        out = model(batch_x)  # torch.Size([128,10])
        out2 = m(out)
        # train_out_prob.append(out2)
        # print(type(out2))
        # 获取损失
        loss = loss_func(out2, batch_y.float())
        # print(type(loss)):<class 'torch.Tensor'>
        # 使用优化器优化损失
        opt.zero_grad()  # 清空上一步残余更新参数值
        loss.backward()  # 误差反向传播，计算参数更新值
        opt.step()  # 将参数更新值施加到net的parmeters上
        #train_losses.append(loss)
        # if i%5 == 0:
        # loss_count.append(loss)
        # print('{}:\t'.format(i), loss.item())
        # torch.save(model,'D:/Liuliwen/MLDF')

    model.eval()  # 设置模型为评估/测试模式
    falselabels = 0
    num_have_valided = 0
    predict_prob_valid = np.zeros((valid_size, num_labels))
    label_true_valid = np.zeros((valid_size, num_labels))
    pre_label_valid = np.zeros((valid_size, num_labels))
    for i, (data, target) in enumerate(valid_loader):
        # print(data)
        f = num_have_valided
        data = Variable(data)
        target = Variable(target)
        output = model(data)
        valid_probability = m(output)
        num_have_valided = f + len(target)
        valid_loss_torch = loss_func(valid_probability, target.float())
        valid_loss = valid_loss_torch.detach().cpu().numpy()
        valid_losses.append(valid_loss)
        # predict_label_valid = label_predict(valid_probability)
        # falselabels = falselabels + false_labels(target, predict_label_valid)
        # print (falselabels)
        label_true_valid[f:num_have_valided, :] = target.cpu().numpy()
        # pre_label_valid[f:num_have_valided, :] = predict_label_valid
        predict_prob_valid[f:num_have_valided, :] = valid_probability.detach().cpu().numpy()
    # print(type(train_losses))
    #train_loss = torch.mean(torch.stack(train_losses))
    #valid_loss = torch.mean(torch.stack(valid_losses))
    valid_loss = np.mean(valid_losses)
    # hamming_loss_valid = falselabels / (valid_size * 5)
    #avg_train_losses.append(train_loss)
    #avg_valid_losses.append(valid_loss)
    # valid_hammingloss.append(hamming_loss_valid)

    #train_losses = []
    valid_losses = []

    early_stopping(valid_loss, model)
    # print(valid_hammingloss)
    # early_stopping(hamming_loss_valid, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break
end = time.time()
train_time = end - start
print("训练结束\n epoch次数：%d" % (epoch))
print("训练时间：%s Seconds" % (train_time))
torch.save(model.state_dict(),"CNN_CBAM.pt")
#thresholds = threshold_desicion(predict_prob_valid, label_true_valid)
row0 = ['macro-acc','macro-f1','macro-AUC','one-error','ranking_loss','ave_precision','coverage','SNR','epoch','train_time']
#row0 = ['accurancy','training time']
workbook = xlwt.Workbook(encoding = 'utf-8')
performance_sheet = workbook.add_sheet('result')
model.eval()
with torch.no_grad():
    count_test_file = 0
    for test_snr in test_snr_val:
        print("测试数据集SNR：",test_snr)
        falselabels = 0
        num_have_tested = 0
        root2 = r"image_test/images_snr_{}.txt".format(test_snr)
        count_test_file += 1
        my_test_datasets = Mydata_sets(root2, transform=transforms)

        test_loader = Data.DataLoader(
            dataset=my_test_datasets,
            batch_size=BATCH_SIZE,
            shuffle=True)
        num_test = len(my_test_datasets)
        predict_prob = np.zeros((num_test, num_labels))
        label_true = np.zeros((num_test, num_labels))
        pre_label = np.zeros((num_test, num_labels))
        for x, y in test_loader:
            # print(len(y))
            test_x = Variable(x)
            test_y = Variable(y)
            # print(test_y.shape)
            a = num_have_tested
            num_have_tested = num_have_tested + len(test_y)
            out = model(test_x)
            out_probability = m(out)
            out_probability = out_probability.detach().cpu().numpy()
            # 得到预测标签
            predict_label = label_predict(out_probability, 0.5)
            falselabels = falselabels + false_labels(test_y, predict_label)
            # print (falselabels)
            label_true[a:num_have_tested, :] =y.cpu().numpy()
            pre_label[a:num_have_tested, :] = predict_label
            predict_prob[a:num_have_tested, :] = out_probability

        hamming_loss = falselabels / (num_test * 5)
        one_error = compute_one_error(predict_prob, label_true)
        ranking_loss = compute_ranking_loss(predict_prob, label_true)
        macro_auc = compute_macro_auc(predict_prob, label_true)
        macro_f1 = compute_macro_f1(pre_label, label_true)
        macro_acc = 1 - hamming_loss
        coverage = compute_coverage(predict_prob, label_true)
        ave_precision = compute_average_precision(predict_prob, label_true)
        print('Final test results:\t')
        print('macro-acc:\t', macro_acc)
        print('macro_f1:\t', macro_f1)
        print('macro-AUC:\t', macro_auc)
        print('one-error:\t', one_error)
        print('ranking_loss:\t', ranking_loss)
        print('ave_precision:\t', ave_precision)
        print('coverage:\t', coverage)
        performance_sheet.write(count_test_file, 0, macro_acc)
        performance_sheet.write(count_test_file, 1, macro_f1)
        performance_sheet.write(count_test_file, 2, macro_auc)
        performance_sheet.write(count_test_file, 3, one_error)
        performance_sheet.write(count_test_file, 4, ranking_loss)
        performance_sheet.write(count_test_file, 5, ave_precision)
        performance_sheet.write(count_test_file, 6, coverage)
        performance_sheet.write(count_test_file, 7, test_snr)
        performance_sheet.write(count_test_file, 8, epoch)
        performance_sheet.write(count_test_file, 9, train_time)
    filename = 'TF_results_CNN_CBAM.xls'
    workbook.save(filename)



'''
epoch= 1
Validation loss decreased (inf --> 0.544322).  Saving model ...
epoch= 2
Validation loss decreased (0.544322 --> 0.523809).  Saving model ...
epoch= 3
Validation loss decreased (0.523809 --> 0.516637).  Saving model ...
epoch= 4
Validation loss decreased (0.516637 --> 0.505723).  Saving model ...
epoch= 5
Validation loss decreased (0.505723 --> 0.502045).  Saving model ...
epoch= 6
Validation loss decreased (0.502045 --> 0.494442).  Saving model ...
epoch= 7
Validation loss decreased (0.494442 --> 0.491454).  Saving model ...
epoch= 8
Validation loss decreased (0.491454 --> 0.490940).  Saving model ...
epoch= 9
Validation loss decreased (0.490940 --> 0.488124).  Saving model ...
epoch= 10
Validation loss decreased (0.488124 --> 0.486255).  Saving model ...
epoch= 11
Validation loss decreased (0.486255 --> 0.474914).  Saving model ...
epoch= 12
Validation loss decreased (0.474914 --> 0.473783).  Saving model ...
epoch= 13
EarlyStopping counter: 1 out of 30
epoch= 14
EarlyStopping counter: 2 out of 30
epoch= 15
EarlyStopping counter: 3 out of 30
epoch= 16
EarlyStopping counter: 4 out of 30
epoch= 17
EarlyStopping counter: 5 out of 30
epoch= 18
EarlyStopping counter: 6 out of 30
epoch= 19
EarlyStopping counter: 7 out of 30
epoch= 20
EarlyStopping counter: 8 out of 30
epoch= 21
EarlyStopping counter: 9 out of 30
epoch= 22
EarlyStopping counter: 10 out of 30
epoch= 23
EarlyStopping counter: 11 out of 30
epoch= 24
EarlyStopping counter: 12 out of 30
epoch= 25
EarlyStopping counter: 13 out of 30
epoch= 26
EarlyStopping counter: 14 out of 30
epoch= 27
EarlyStopping counter: 15 out of 30
epoch= 28
EarlyStopping counter: 16 out of 30
epoch= 29
Final test results:
macro-acc:       0.5316129032258065
macro_f1:        0.5182654131405983
macro-AUC:       0.5342083333333332
one-error:       0.45161290322580644
ranking_loss:    0.47607526881720413
ave_precision:   0.681917562724014
coverage:        0.635483870967742
测试数据集SNR： -15
Final test results:
macro-acc:       0.5464516129032257
macro_f1:        0.5453541075909412
macro-AUC:       0.57125
one-error:       0.45483870967741935
ranking_loss:    0.4489247311827957
ave_precision:   0.6955779569892465
coverage:        0.604516129032258
测试数据集SNR： -10
Final test results:
macro-acc:       0.547741935483871
macro_f1:        0.5345894837590286
macro-AUC:       0.5619666666666667
one-error:       0.46774193548387094
ranking_loss:    0.44838709677419375
ave_precision:   0.6934318996415767
coverage:        0.6006451612903225
测试数据集SNR： -5
Final test results:
macro-acc:       0.5774193548387097
macro_f1:        0.585049596122452
macro-AUC:       0.6117041666666667
one-error:       0.29354838709677417
ranking_loss:    0.36639784946236564
ave_precision:   0.7576478494623651
coverage:        0.5748387096774193
测试数据集SNR： 0
Final test results:
macro-acc:       0.7541935483870967
macro_f1:        0.7569795243883475
macro-AUC:       0.8325416666666665
one-error:       0.1064516129032258
ranking_loss:    0.16344086021505372
ave_precision:   0.8930734767025089
coverage:        0.44129032258064516
测试数据集SNR： 5
Final test results:
macro-acc:       0.8587096774193548
macro_f1:        0.8653248109491345
macro-AUC:       0.9180250000000001
one-error:       0.035483870967741936
ranking_loss:    0.06801075268817205
ave_precision:   0.9543503584229385
coverage:        0.37483870967741933
测试数据集SNR： 10
Final test results:
macro-acc:       0.8864516129032258
macro_f1:        0.8861364239061815
macro-AUC:       0.9389916666666668
one-error:       0.01935483870967742
ranking_loss:    0.04865591397849459
ave_precision:   0.9683064516129029
coverage:        0.3612903225806452
测试数据集SNR： 15
Final test results:
macro-acc:       0.8793548387096775
macro_f1:        0.878465417466253
macro-AUC:       0.9332125000000001
one-error:       0.016129032258064516
ranking_loss:    0.04677419354838708
ave_precision:   0.972674731182795
coverage:        0.35935483870967744
测试数据集SNR： 20
Final test results:
macro-acc:       0.8774193548387097
macro_f1:        0.8737093055127474
macro-AUC:       0.9342166666666666
one-error:       0.02903225806451613
ranking_loss:    0.05672043010752686
ave_precision:   0.9636021505376345
coverage:        0.36580645161290326
'''
