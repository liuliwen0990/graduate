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
    [
     transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)
'''
import random
manualSeed = 1
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 这么设置使用确定性算法，如果代码中有算法cuda没有确定性实现，则会报错，可以验证代码中有没有cuda没有确定性实现的代码
#torch.use_deterministic_algorithms(True)
# 这么设置使用确定性算法，如果代码中有算法cuda没有确定性实现，也不会报错
torch.use_deterministic_algorithms(True, warn_only=True)

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
'''

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
'''
def label_predict(output, threshold=0.5,num_labels = 1):
    num_examples = len(output)
    predict_label = np.zeros((num_examples, num_labels))
    for i in range(0, num_examples):
        if output[i] >= threshold:
            predict_label[i,0] = 1

    return predict_label
'''
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
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


class MLCNN(nn.Module):

    def __init__(self):
        super(MLCNN, self).__init__()

        # in_channels输入数据通道数
        # padding:第一个数是高度，第二个是宽度
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.attention1 = CBAMBlock(channel=64,reduction=16,kernel_size=5)

        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.attention2 = CBAMBlock(channel=32,reduction=16,kernel_size=5)

        self.pool2 = nn.MaxPool2d(2,stride=2)

        self.dropout1 = nn.Dropout(p=0.3)  # dropout训练
        self.dropout2 = nn.Dropout(p=0.3)  # dropout训练

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1, padding=(1,1), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.attention3 = CBAMBlock(channel=16,reduction=16,kernel_size=5)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        # 32*15*15
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        #16*16*16
        self.attention4 = CBAMBlock(channel=8, reduction=8, kernel_size=7)
        self.mlp1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        #self.dropout3 = nn.Dropout(0.2)
        self.mlp2 = nn.Sequential(
        nn.Linear(128, 5),
        )

    def forward(self, x):
        """
        input: N * 3 * image_size * image_size
        output: N * num_classes
        """
        x = self.conv1(x)
        #x = self.attention1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        #x = self.attention2(x)
        x = self.pool2(x)

        x = self.dropout1(x)
        x = self.conv3(x)
        #x = self.attention3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        #x = self.attention4(x)
        x = self.pool4(x)
        x = self.dropout2(x)
        
        #print(x.shape)
        # view(x.size(0), -1): change tensor size from (N ,H , W) to (N, H*W)
        x = self.mlp1(x.view(x.size(0), -1))
        #x = self.dropout3(x)
        x = self.mlp2(x)

        return x

def threshold_desicion(y_prob, y_label):
    thresholds = []
    num_samples,num_label = y_label.shape

    threshold = [0]*num_label
    eplison = np.arange(0, 1, 0.05)

    for i in range(num_label):
        F = np.zeros((1,len(eplison)))

        y_label_list = list(y_label[:,i])
        true_1_sum = sum(y_label_list)
        #print(true_1_sum)
        #print(y_prob[:, i])
        for j in range(len(eplison)):
            predict_label = label_predict(y_prob[:,i],eplison[j],1)
            predict_1_sum = sum(predict_label)
            #print("predict_1_sum=", predict_1_sum)
            count = 0
            for k in range(num_samples):
                count += predict_label[k,0] * y_label[k,i]

            #print("count=",count)
            F[0,j] = 2* count/(predict_1_sum+true_1_sum)

        max_index = np.where(F == np.max(F))
        index = max(max_index[1])
        #print(max_index[1])
        #print(index)
        threshold[i] = eplison[index]
    print(threshold)
    return threshold

def label_predict_new(output, thresholds):
    num_examples, num_labels = output.shape
    predict_label = np.zeros((num_examples, num_labels))
    for i in range(0, num_examples):
        for j in range(0, num_labels):
            if output[i, j] >= thresholds[j]:
                predict_label[i, j] = 1
        if not 1 in predict_label[i,:]:
            index = np.where(output[i,:]==max(output[i,:]))
            predict_label[i,index]=1

    return predict_label

#train_snr_val = ['-20','-15','-10','-5','0','5','10','15','20']
train_snr_val = ['-20','-15','-10','-5','0','5','10','15','20']
test_snr_val = ['-20','-15','-10','-5','0','5','10','15','20']

BATCH_SIZE = 8
num_labels = 5
count = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

# 我这里面作弊了
#train_dataset += my_test_datasets

train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True)
valid_loader = Data.DataLoader(
            dataset=my_test_datasets,
            shuffle=True)



model = MLCNN()

model.apply(weight_init)
model = model.to(device)
m = nn.Sigmoid()
loss_func = nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)  # 论文就是0.01,weight_decay=0.01
n_epoch = 100
patience = 10  # 当验证集损失在连续15次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
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
#thresholds = threshold_desicion(predict_prob_valid, label_true_valid)
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
            predict_label = label_predict(out_probability)
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
    import sys
    filename = sys.argv[0] + "without_attention.xls"
    workbook.save(filename)

'''
测试数据集SNR： -20
Final test results:
macro-acc:       0.5180645161290323
macro_f1:        0.5572756400661332
macro-AUC:       0.5271416666666667
one-error:       0.47419354838709676
ranking_loss:    0.48360215053763456
ave_precision:   0.6742965949820782
coverage:        0.6290322580645162
测试数据集SNR： -15
Final test results:
macro-acc:       0.5212903225806451
macro_f1:        0.5494822587117453
macro-AUC:       0.5239333333333334
one-error:       0.47419354838709676
ranking_loss:    0.47607526881720463
ave_precision:   0.6768951612903219
coverage:        0.6225806451612904
测试数据集SNR： -10
Final test results:
macro-acc:       0.5206451612903226
macro_f1:        0.5768123820683357
macro-AUC:       0.5340416666666667
one-error:       0.4032258064516129
ranking_loss:    0.4182795698924733
ave_precision:   0.7144623655913975
coverage:        0.5967741935483871
测试数据集SNR： -5
Final test results:
macro-acc:       0.6077419354838709
macro_f1:        0.6758348289733433
macro-AUC:       0.647375
one-error:       0.26129032258064516
ranking_loss:    0.3322580645161291
ave_precision:   0.7773655913978491
coverage:        0.5574193548387096
测试数据集SNR： 0
Final test results:
macro-acc:       0.7735483870967742
macro_f1:        0.8063919981385288
macro-AUC:       0.8516916666666667
one-error:       0.06451612903225806
ranking_loss:    0.11505376344086021
ave_precision:   0.9266801075268809
coverage:        0.40387096774193554
测试数据集SNR： 5
Final test results:
macro-acc:       0.8606451612903225
macro_f1:        0.8757987909690155
macro-AUC:       0.9257500000000001
one-error:       0.025806451612903226
ranking_loss:    0.060215053763440864
ave_precision:   0.9628673835125442
coverage:        0.36645161290322575
测试数据集SNR： 10
Final test results:
macro-acc:       0.8877419354838709
macro_f1:        0.8991042995909465
macro-AUC:       0.9396666666666664
one-error:       0.02903225806451613
ranking_loss:    0.05403225806451611
ave_precision:   0.9644220430107522
coverage:        0.36322580645161284
测试数据集SNR： 15
Final test results:
macro-acc:       0.8883870967741936
macro_f1:        0.895531071020339
macro-AUC:       0.9426916666666665
one-error:       0.01935483870967742
ranking_loss:    0.04489247311827955
ave_precision:   0.9689381720430108
coverage:        0.35935483870967744
测试数据集SNR： 20
Final test results:
macro-acc:       0.8812903225806452
macro_f1:        0.8900041573299097
macro-AUC:       0.9376083333333334
one-error:       0.02903225806451613
ranking_loss:    0.04838709677419354
ave_precision:   0.9666263440860212
coverage:        0.35870967741935483
'''