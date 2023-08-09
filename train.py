# coding:utf-8
import datetime
import time
import os
from PIL.Image import Image
from torch import optim
from loss.losses import *
from data.dataloader import ISBI_Loader, ISBI_Loadertest
import torch.nn as nn
import torch
from models.EIANet import EIANet
from torchvision import transforms
from torch.nn import functional as F
from evaluation import *
from torchvision import transforms as T
transform = transforms.Compose([
    transforms.ToTensor()])


def train_net(net, device, train_data_path, test_data_path, fold, epochs=200, batch_size=8, lr=0.00001):
    transform = []
    transform.append(T.Resize((256, 256), interpolation=1))
    transform.append(T.ToTensor())
    transform = T.Compose(transform)
    isbi_train_dataset = ISBI_Loader(train_data_path, transform)
    print("train", len(isbi_train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_dataset = ISBI_Loadertest(test_data_path, transform)
    print("test", len(test_dataset))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    criterion3 = structure_loss()

    result = 0

    start_time = time.time()
    for epoch in range(epochs):
        net.train()

        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        JS = 0.  # Jaccard Similarity
        DC = 0.  # Dice Coefficient
        count = 0
        loss_list = []
        losspred_list = []
        loss2_list = []
        loss3_list = []
        loss4_list = []
        loss5_list = []
        for image, label, edge in train_loader:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred, x22_d, x33_d, x44_d, x55_d = net(image)
            x22_l = F.interpolate(label, scale_factor=0.5, mode='bilinear', align_corners=True)
            x33_l = F.interpolate(label, scale_factor=0.25, mode='bilinear', align_corners=True)
            x44_l = F.interpolate(label, scale_factor=0.125, mode='bilinear', align_corners=True)
            x55_l = F.interpolate(label, scale_factor=0.0625, mode='bilinear', align_corners=True)
            loss_pred = criterion3(pred, label)
            loss_2 = criterion3(x22_d, x22_l)
            loss_3 = criterion3(x33_d, x33_l)
            loss_4 = criterion3(x44_d, x44_l)
            loss_5 = criterion3(x55_d, x55_l)
            loss = alpha * loss_pred + beta * (loss_2 + loss_3 + loss_4 + loss_5)
            loss_list.append(loss.detach().cpu().numpy())
            losspred_list.append(loss_pred.detach().cpu().numpy())
            loss2_list.append(loss_2.detach().cpu().numpy())
            loss3_list.append(loss_3.detach().cpu().numpy())
            loss4_list.append(loss_4.detach().cpu().numpy())
            loss5_list.append(loss_5.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            sig = torch.nn.Sigmoid()
            pred = sig(pred)
            # print(pred.shape)
            acc += get_accuracy(pred, label)
            SE += get_sensitivity(pred, label)
            SP += get_specificity(pred, label)
            PC += get_precision(pred, label)
            F1 += get_F1(pred, label)
            JS += get_JS(pred, label)
            DC += get_DC(pred, label)
            count += 1
        acc = acc / count
        SE = SE / count
        SP = SP / count
        PC = PC / count
        F1 = F1 / count
        JS = JS / count
        DC = DC / count
        score = JS + DC

        et = time.time() - start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        print(et, 'Train Epoch:{}'.format(epoch))
        print('Loss', np.mean(loss_list))
        print("acc=" + str(acc) + ", SE=" + str(SE) + ",SP=" + str(SP) + ",PC=" + str(PC) + ",F1=" + str(
            F1) + ",JS=" + str(JS) + ",DC=" + str(DC) + ",Score=" + str(score))
        if epoch > 0:
            # net.to(device)
            sig = torch.nn.Sigmoid()
            net.eval()
            with torch.no_grad():
                # when in test stage, no grad
                acc = 0.  # Accuracy
                SE = 0.  # Sensitivity (Recall)
                SP = 0.  # Specificity
                PC = 0.  # Precision
                F1 = 0.  # F1 Score
                JS = 0.  # Jaccard Similarity
                DC = 0.  # Dice Coefficient
                count = 0
                alpha=1
                beta=0.5
                criterion3 = structure_loss()
                loss_list = []
                losspred_list = []
                loss2_list = []
                loss3_list = []
                loss4_list = []
                loss5_list = []
                for image, label, edge in test_loader:
                    image = image.to(device=device, dtype=torch.float32)
                    label = label.to(device=device, dtype=torch.float32)
                    pred, x22_d, x33_d, x44_d, x55_d = net(image)
                    x22_l = F.interpolate(label, scale_factor=0.5, mode='bilinear', align_corners=True)
                    x33_l = F.interpolate(label, scale_factor=0.25, mode='bilinear', align_corners=True)
                    x44_l = F.interpolate(label, scale_factor=0.125, mode='bilinear', align_corners=True)
                    x55_l = F.interpolate(label, scale_factor=0.0625, mode='bilinear', align_corners=True)
                    loss_pred = criterion3(pred, label)
                    loss_2 = criterion3(x22_d, x22_l)
                    loss_3 = criterion3(x33_d, x33_l)
                    loss_4 = criterion3(x44_d, x44_l)
                    loss_5 = criterion3(x55_d, x55_l)
                    loss = alpha*loss_pred + beta*(loss_2 + loss_3 + loss_4 + loss_5)
                    loss_list.append(loss.detach().cpu().numpy())
                    losspred_list.append(loss_pred.detach().cpu().numpy())
                    loss2_list.append(loss_2.detach().cpu().numpy())
                    loss3_list.append(loss_3.detach().cpu().numpy())
                    loss4_list.append(loss_4.detach().cpu().numpy())
                    loss5_list.append(loss_5.detach().cpu().numpy())
                    sig = torch.nn.Sigmoid()
                    pred = sig(pred)
                    # print(pred.shape)
                    acc += get_accuracy(pred, label)
                    SE += get_sensitivity(pred, label)
                    SP += get_specificity(pred, label)
                    PC += get_precision(pred, label)
                    F1 += get_F1(pred, label)
                    JS += get_JS(pred, label)
                    DC += get_DC(pred, label)
                    count += 1
                acc = acc / count
                SE = SE / count
                SP = SP / count
                PC = PC / count
                F1 = F1 / count
                JS = JS / count
                DC = DC / count
                score = JS + DC
                print(et, 'Test Epoch:{}'.format(epoch))
                print('Loss', np.mean(loss_list))

            print("acc=" + str(acc) + ", SE=" + str(SE) + ",SP=" + str(SP) + ",PC=" + str(PC) + ",F1=" + str(
                F1) + ",JS=" + str(JS) + ",DC=" + str(DC) + ",Score=" + str(score))
            if result < DC:
                result = DC
                torch.save(net.state_dict(), './result/modify/result' + str(fold) + '.pth')
                with open("./result/modify/result" + str(fold) + ".csv", "a") as w:
                    w.write("acc=" + str(acc) + ", SE=" + str(SE) + ",SP=" + str(SP) + ",PC=" + str(PC) + ",F1=" + str(
                        F1) + ",JS=" + str(JS) + ",DC=" + str(DC) + ",Score=" + str(score) + "\n")
    if result > 0.88:
        print("attention! The DC more than 88% appeared! ", result)
    else:
        print("The best is ", result)
    return result


if __name__ == "__main__":
    import os
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DSC_list = []

    for i in range(5):
        DSC = 0
        net = EIANet(n_channels=1, n_classes=1)
        net.to(device=device)
        data_path = ""
        test_data_path = ""
        path = "./result/result" + str(i) + ".csv"
        print(os.path.exists(path))
        if os.path.exists(path):
            os.remove(path)
        DSC = train_net(net, device, data_path, test_data_path, i, epochs=100, batch_size=32)
        DSC_list.append(DSC)
    DSC_list = np.array(DSC_list)
    print(DSC_list.mean())





