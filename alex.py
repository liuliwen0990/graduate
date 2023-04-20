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

class MLCNN(nn.Module):

    def __init__(self):
        super(MLCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(11, 11), stride=2, padding=(2, 2), bias=True),
            nn.ReLU(),
        )

        self.pool1 = nn.MaxPool2d(3, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(5, 5), stride=1, padding=2, bias=True),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(3,stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=1, padding=(1,1), bias=True),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=(1,1), bias=True),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=(1,1), bias=True),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(3, stride=2)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.mlp1 = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(),
        )

        self.mlp2 = nn.Sequential(
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 5),
        )

    def forward(self, x):
        """
        input: N * 3 * image_size * image_size
        output: N * num_classes
        """
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.mlp1(x)
        x = self.dropout2(x)
        x = self.mlp2(x)

        return x


#train_snr_val = ['-20','-15','-10','-5','0','5','10','15','20']
train_snr_val = ['-20','-15','-10','-5','0','5','10','15','20']
test_snr_val = ['-20','-15','-10','-5','0','5','10','15','20']
BATCH_SIZE = 8
num_labels = 5
count = 0
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
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



model = MLCNN()

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
Final test results:
macro-acc:       0.5258064516129033
macro_f1:        0.5544088850394602
macro-AUC:       0.5371416666666666
one-error:       0.4870967741935484
ranking_loss:    0.47365591397849466
ave_precision:   0.6792338709677412
coverage:        0.6167741935483871
测试数据集SNR： -15
Final test results:
macro-acc:       0.5032258064516129
macro_f1:        0.5486107561305497
macro-AUC:       0.5278833333333334
one-error:       0.47096774193548385
ranking_loss:    0.47043010752688164
ave_precision:   0.6788440860215051
coverage:        0.6187096774193549
测试数据集SNR： -10
Final test results:
macro-acc:       0.5212903225806451
macro_f1:        0.5641693827278729
macro-AUC:       0.5325000000000001
one-error:       0.42258064516129035
ranking_loss:    0.46344086021505393
ave_precision:   0.6905107526881715
coverage:        0.6245161290322582
测试数据集SNR： -5
Final test results:
macro-acc:       0.6025806451612903
macro_f1:        0.6426919516208984
macro-AUC:       0.6509249999999999
one-error:       0.3032258064516129
ranking_loss:    0.32876344086021503
ave_precision:   0.7802777777777774
coverage:        0.5354838709677419
测试数据集SNR： 0
Final test results:
macro-acc:       0.7593548387096775
macro_f1:        0.7833770483352799
macro-AUC:       0.8304583333333333
one-error:       0.09032258064516129
ranking_loss:    0.1567204301075269
ave_precision:   0.9065232974910389
coverage:        0.42838709677419357
测试数据集SNR： 5
Final test results:
macro-acc:       0.8541935483870968
macro_f1:        0.8674885611039087
macro-AUC:       0.9162916666666666
one-error:       0.02258064516129032
ranking_loss:    0.06532258064516129
ave_precision:   0.9617249103942648
coverage:        0.3690322580645161
测试数据集SNR： 10
Final test results:
macro-acc:       0.8825806451612903
macro_f1:        0.8950638798261379
macro-AUC:       0.9354833333333333
one-error:       0.03225806451612903
ranking_loss:    0.05779569892473119
ave_precision:   0.9601344086021502
coverage:        0.36709677419354836
测试数据集SNR： 15
Final test results:
macro-acc:       0.8729032258064516
macro_f1:        0.8846031548243618
macro-AUC:       0.9350249999999999
one-error:       0.02903225806451613
ranking_loss:    0.06129032258064514
ave_precision:   0.9591756272401428
coverage:        0.3735483870967742
测试数据集SNR： 20
Final test results:
macro-acc:       0.8858064516129032
macro_f1:        0.8960117426851921
macro-AUC:       0.9388916666666667
one-error:       0.02903225806451613
ranking_loss:    0.04838709677419352
ave_precision:   0.9654211469534051
coverage:        0.35870967741935483
'''

'''
训练时间：9396.011615753174 Seconds
测试数据集SNR： -20
Final test results:
macro-acc:       0.5019354838709678
macro_f1:        0.5156034664926417
macro-AUC:       0.5122083333333334
one-error:       0.4483870967741935
ranking_loss:    0.4844086021505376
ave_precision:   0.675430107526881
coverage:        0.636774193548387
测试数据集SNR： -15
Final test results:
macro-acc:       0.5109677419354839
macro_f1:        0.49901064676160206
macro-AUC:       0.53
one-error:       0.5032258064516129
ranking_loss:    0.48010752688172065
ave_precision:   0.6716890681003584
coverage:        0.6277419354838709
测试数据集SNR： -10
Final test results:
macro-acc:       0.5129032258064516
macro_f1:        0.5437827404317899
macro-AUC:       0.5146833333333334
one-error:       0.4483870967741935
ranking_loss:    0.45725806451612894
ave_precision:   0.6900672043010752
coverage:        0.6212903225806452
测试数据集SNR： -5
Final test results:
macro-acc:       0.6225806451612903
macro_f1:        0.6509496398785874
macro-AUC:       0.6629250000000001
one-error:       0.27419354838709675
ranking_loss:    0.3153225806451612
ave_precision:   0.796509856630824
coverage:        0.5290322580645161
测试数据集SNR： 0
Final test results:
macro-acc:       0.7619354838709678
macro_f1:        0.7847267249492282
macro-AUC:       0.8213416666666667
one-error:       0.08709677419354839
ranking_loss:    0.14489247311827946
ave_precision:   0.9074686379928314
coverage:        0.4290322580645161
测试数据集SNR： 5
Final test results:
macro-acc:       0.8606451612903225
macro_f1:        0.8698095618358923
macro-AUC:       0.9221083333333333
one-error:       0.035483870967741936
ranking_loss:    0.06774193548387093
ave_precision:   0.9564247311827951
coverage:        0.37225806451612903
测试数据集SNR： 10
Final test results:
macro-acc:       0.8941935483870967
macro_f1:        0.9016916475986532
macro-AUC:       0.9373999999999999
one-error:       0.035483870967741936
ranking_loss:    0.05698924731182796
ave_precision:   0.9605241935483867
coverage:        0.36580645161290326
测试数据集SNR： 15
Final test results:
macro-acc:       0.8916129032258064
macro_f1:        0.8942735533310205
macro-AUC:       0.9401916666666666
one-error:       0.02903225806451613
ranking_loss:    0.04811827956989247
ave_precision:   0.9710573476702503
coverage:        0.3567741935483871
测试数据集SNR： 20
Final test results:
macro-acc:       0.8754838709677419
macro_f1:        0.8804347127722425
macro-AUC:       0.9298500000000001
one-error:       0.025806451612903226
ranking_loss:    0.05134408602150533
ave_precision:   0.9642114695340502
coverage:        0.3651612903225806
'''