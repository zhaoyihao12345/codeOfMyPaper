import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import LiverDataset, Val_Dataset
import time
import cv2
from Visualizer import Visualizer
import scipy.misc as misc
import random
import imageio
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
from unet_sSE import Unet_1
from unet_cSE import Unet_2
from unet_csSE import Unet_12
from unet_CA import Unet_CA
from unet_PA import Unet_PA
from FCDenseNet import FCDenseNet56, FCDenseNet67, FCDenseNet103
from nets.unet_spp import Unet_spp
from nets.unet_CA_spp import Unet_CA_spp
# from nets.Unet_DenseBlock import DenseUnet
from nets.unet_DAC import Unet_DAC
# from nets.Unet_DenseBlock_new import DenseUnet1
from nets.Unet_DenseBlock_DAC import DenseUnet_DAC
from nets.Unet_DenseBlock_CA import DenseUnet_CA
from nets.Unet_DenseBlock_sSE import DenseUnet_sSE
from nets.unet_gate import Unet_gate
from nets.unet_global_gate import Unet_global_gate
from nets.unet_sSE1 import Unet_sSE1
from nets.unet_cSE1 import Unet_cSE1
# from nets.unet_CA1 import Unet_CA1
from nets.DenseUnet_cSE_globalgate import DenseUnet_cSE_globalgate
from nets.DenseUnet_globalgate import DenseUnet_globalgate
from nets.DenseUnet_cSE import DenseUnet_cSE
from nets.DenseUnet_cSE_nomiddle import DenseUnet_cSE_nomiddle
# from nets.Unet_DenseBlock import DenseUnet
from nets.Unet_DenseBlock1 import DenseUnet
from nets.FulDenseNet import FulDenseUnet
from nets.unet_cSE1_gate1 import UnetcSE1gate1
from loss import MulticlassDiceLoss_CrossEntropyLoss
from nets.unet_global_gate1 import Unet_global_gate1
from nets.unet_cSE_sSE import Unet_cSE_sSE
from nets.unet_BAM import Unet_BAM
from nets.unet_CBAM import Unet_CBAM
from nets.unet_CA_New import Unet_CA_New
from nets.unet_CANew_BAM import Unet_CANew_BAM

from nets.unet_global_conv import Unet_global_conv
from nets.unet_mul_input import Unet_mul_input
from nets.unet_BAM_gate import Unet_BAM_gate
from nets.unet_DAC import Unet_DAC
from loss import MulticlassDiceLoss
from nets.cenet import CE_Net_
from nets.unet_SGE import Unet_SGE
from nets.cenet_se import se_CE_Net_
from nets.unet_GC import Unet_GC
from nets.unet_GC1 import Unet_GC1
from nets.unet_ResNet34 import Unet_ResNet34
from nets.unet_ResNet34_DessASPP import Unet_ResNet34_DenseASPP
from nets.unet_GC_DessASPP import Unet_GC_DenseASPP
from nets.unet_GC_ResNet34 import Unet_GC_ResNet34
from nets.unet_Mul_ResNet34 import Unet_Mul_ResNet34
from nets.DeepDisc import DeepDisc
from nets.unet_ResNet34_gate import Unet_ResNet34_gate
from nets.unet_ResNet34_cSE import Unet_ResNet34_cSE
from nets.unet_ResNet34_csSE import Unet_ResNet34_csSE
from nets.unet_ResNet34_1X1conv import Unet_ResNet34_1X1conv
from nets.unet_ResNet34_cSE_down16 import Unet_ResNet34_cSE_down16
from nets.unet_ResNet34_RAU import Unet_ResNet34_RAU
from nets.unet_ResNet34_cSE_refine import Unet_ResNet34_cSE_refine
from nets.unet_ResNet34_cSE_aspp import Unet_ResNet34_cSE_aspp
from nets.unet_ResNet50 import Unet_ResNet50
from nets.unet_ResNet18 import Unet_ResNet18
from nets.unet_ResNet18_cSE import Unet_ResNet18_cSE
from nets.unet_ResNet34_cSE_deep import Unet_ResNet34_cSE_deep
from nets.unet_ResNet34_cSE_nopool_deep import Unet_ResNet34_cSE_nopool_deep
from nets.unet_ResNet34_cSE_deep_aspp import Unet_ResNet34_cSE_deep_aspp
from nets.unet_ResNet34_cSE_deep_gram import Unet_ResNet34_cSE_deep_gram
from nets.unet_ResNet34_ECA import Unet_ResNet34_ECA
from nets.unet_ResNet34_cSE_last import Unet_ResNet34_cSE_last
from nets.unet_ResNet34_cSE_full import Unet_ResNet34_cSE_full
from nets.unet_ResNet34_cSE_5 import Unet_ResNet34_cSE_5
from nets.unet_ResNet34_cSE_skip import Unet_ResNet34_cSE_skip

from nets.unet_ResNet34_GC import Unet_ResNet34_GC
from nets.unet_ResNet34_cSE_spp import Unet_ResNet34_cSE_spp
from nets.fcn import FCN
from nets.unet_ResNet34_cSE_decoder import Unet_ResNet34_cSE_decoder
from nets.Deeplab import DeepLabV3Plus
from nets.unet_ResNet34_cSE_CA import Unet_ResNet34_cSE_CA
from skimage.measure import label, regionprops
import scipy

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# device = torch.device("cuda" )
x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()

# 参数解析
parse = argparse.ArgumentParser()


# 计算Dice系数
# def soft_dice_coeff(input, target):
#     N = target.size(0)
#     smooth = 1
#     input_flat = input.view(N, -1)
#     target_flat = target.view(N, -1)
#     intersection = input_flat * target_flat
#     dice = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
#
#     return dice
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


best_Dice = 0


def BW_img(input):
    binary = input
    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))


def train_model(model, criterion, optimizer, dataload, dataloader_val, num_epochs=60):
    # run the Visdom
    # viz = Visualizer()

    loss_list = []
    for epoch in range(1, num_epochs + 1):
        # start = time.asctime()
        # start = time.clock()
        start = time.time()

        # if epoch > 10:
        #     for name, value in model.named_parameters():
        #         if name in no_grad:
        #             value.requires_grad = True
        #         else:
        #             value.requires_grad = True

        print(start)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        # dice = 0

        global best_Dice
        # Jaccard = 0
        num_classes = 2
        val_confusion = np.zeros((num_classes, 3))

        i = 1
        for x, y in dataload:
            print('**** image ****')
            print(i)
            inputs = x.to(device)
            labels = y.to(device)
            # inputs = x
            # labels = y
            # shape1 = x.shape
            # shape2 = y.shape
            # print('inputs_shape:', x.shape)  #[6,3,224,224]
            # print('input type:', type(inputs))  #torch.Tensor
            # inputs = cv2.linearPolar(inputs.numpy(), (112, 112), 112, cv2.WARP_FILL_OUTLIERS)  #[6,3]
            # labels = cv2.linearPolar(labels.numpy(), (112, 112), 112, cv2.WARP_FILL_OUTLIERS)  #[6,1]
            # print('inputs_PT_size:', inputs.shape)
            # print('labels_PT_size:', labels.shape)
            # inputs = np.reshape(inputs, shape1)
            # labels = np.reshape(labels, shape2)
            # inputs = torch.tensor(inputs).to(device)
            # labels = torch.tensor(labels).to(device)

            print(torch.cuda.current_device())
            step += 1
            optimizer.zero_grad()
            # outputs = model(inputs)
            # ave_out, side_6, side_7, side_8, side_9, final_output = model(inputs)
            # ave_out, side_6, side_7, side_8, side_9, final_output = model(inputs)
            model1 = model.train()
            ave_out, side_6, side_7, side_8, final_output = model1(inputs)
            # final_output = model1(inputs)

            # labels = torch.squeeze(labels) #BCE

            # print('output_size:', outputs.size())
            # print('outputs:', outputs.dtype)
            # print('label_size:', labels.size())
            # print('label.dtype:', outputs.dtype)

            # labels = torch.squeeze(labels)  # 交叉熵要求是3D，不Squeeze的话是4D
            # print('label1_size:', labels.size())

            # 深度监督
            loss = criterion(ave_out, labels.long())
            loss += criterion(side_6, labels.long())
            loss += criterion(side_7, labels.long())
            loss += criterion(side_8, labels.long())
            loss += criterion(final_output, labels.long())

            # loss = criterion(final_output, labels.long())  # 交叉熵要求target是Long类型，BCE不用转化为Long类型
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            i += 1
            # print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
            print("train_loss:%0.5f" % (loss.item()))
            print('=' * 10)

        loss_list.append(epoch_loss)

        # 验证
        for x, y in dataloader_val:
            print('**** val ****')

            val_inputs = x.to(device)
            val_labels = y.to(device)
            print('val_labels_ori shape:', val_labels.shape)  # [1,1,224,224]
            # outputs_val = model(val_inputs)
            # ave_out, side_6, side_7, side_8, side_9, outputs_val = model(val_inputs)
            model1.eval()
            ave_out, side_6, side_7, side_8, outputs_val = model1(val_inputs)
            # outputs_val = model1(val_inputs)
            # 深度监督
            val_preds = torch.max(outputs_val, 1)[1]  # [B,H,W] 元素大小不超过<=2
            val_preds = torch.squeeze(val_preds).cpu().numpy()

            # 极坐标转换
            # val_labels = torch.squeeze(val_labels).cpu().numpy()
            # print('shape:', val_preds.shape)
            # val_preds = cv2.linearPolar(val_preds, (112, 112), 112, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
            # print('val_preds shape:', val_preds.shape)  # [224,224]
            # val_labels = cv2.linearPolar(val_labels, (112, 112), 112, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
            # val_labels = np.reshape(val_labels, (1, 1, 224, 224))
            # print('val_labels shape:', val_labels.shape)  # [224,224]
            # val_labels = torch.tensor(val_labels).to(device)

            val_preds = BW_img(val_preds)
            # val_preds = val_preds.reshape(val_preds.shape[0], 1, val_preds.shape[1], val_preds.shape[2])
            val_preds = val_preds.reshape(1, 1, val_preds.shape[0], val_preds.shape[1])

            for i in range(num_classes):
                val_labels_mask = val_labels == i
                val_preds_mask = val_preds == i
                val_labels = val_labels.long()
                TP = np.sum((val_preds == val_labels)[val_labels_mask].data.cpu().numpy())

                val_confusion[i, 0] += TP
                val_confusion[i, 1] += np.sum((val_labels == val_labels)[val_labels_mask].data.cpu().numpy()) - TP
                # val_confusion[i, 2] += np.sum((val_preds == val_preds)[val_preds_mask].data.cpu().numpy()) - TP
                val_confusion[i, 2] += np.sum((val_preds == val_preds)[val_preds_mask]) - TP

        for i in range(1, num_classes):
            TP, FP, FN = val_confusion[i]
            # print(TP + FP, FN)
            dice = (2 * TP) / (FN + FP + 2 * TP)
            Jaccard = TP / (FN + TP + FP)
            pre = TP / (TP + FP)
            recall = TP / (TP + FN)
            print('dice is %0.4f,Jaccard is %0.4f' % (dice, Jaccard))
            # print('dice is %0.4f,Jaccard is %0.4f,pre is %0.4f,recall is %0.4f' % (dice, Jaccard, pre, recall))

        if dice > best_Dice:
            TempBestDicePath = os.path.join(
                'canshu//unetCA_epoch%d_Best_Dice_%.4f__Best_Jaccard_%.4f.pth' % (epoch, dice, Jaccard))
            if os.path.exists(TempBestDicePath):
                os.remove(TempBestDicePath)

            best_Dice = dice
            Best_Jaccard = Jaccard
            torch.save(model.state_dict(),
                       'canshu/ours_eval_shuffle_Pre_cov16_decoder_Aspp6_noPT_Deep_224_cup(refuge)_epoch%d_Best_Dice_%.4f__Best_Jaccard_%.4f__pre_%.4f__recall_%.4f.pth' % (
                           epoch, best_Dice, Best_Jaccard, pre, recall))

    # show the original images, predication and ground truth on the visdom.

    # show_image = x * 255.
    # viz.img(name='images', img_=x[0, :, :, :])
    # viz.img(name='labels', img_=y[0, :, :, :])
    # viz.img(name='prediction', img_=labels[0, :, :, :])

    # end = time.clock()
    end = time.time()
    print("run time:", str(end - start))
    # print(time.time())
    print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    # print('Best_Dice_%.4f__Best_Jaccard_%.4f.pth' % (best_Dice, Best_Jaccard))

    # 绘制loss曲线
    # x = range(1, num_epochs + 1)
    # y = loss_list
    # plt.title('image loss vs. epoches')
    # plt.plot(x, y, 'b-*')
    # plt.ylabel('image loss')
    # plt.xlabel('epoches')
    # # plt.show()
    # plt.savefig("CENet.jpg")

    return model


# 训练模型
def train():
    model = Unet(3, 2).to(device)  # U-Net
    # model = Unet_1(3, 2).to(device)  #sSE
    # model = Unet_2(3, 1).to(device)  #cSE
    # model = Unet_12(3, 1).to(device)  #csSE
    # model = Unet_CA(3, 1).to(device)  # CA
    # model = Unet_PA(3, 1).to(device)  # PA
    # model = FCDenseNet67(2).to(device)  # FC-DenseNet56
    # model = Unet_spp(3, 2).to(device)  # Unet_spp
    # model = Unet_CA_spp(3, 2).to(device)  # Unet_CA_spp
    # model = DenseUnet(3, 2).to(device)  # DenseUNet
    # model = Unet_DAC(3, 2).to(device)  # UNet_DAC
    # model = DenseUnet(3, 2).to(device)  # DenseUNet
    # model = DenseUnet_DAC(3, 2).to(device)  # DenseUNet
    # model = DenseUnet_CA(3, 2).to(device)  # DenseUNet_CA
    # model = DenseUnet_sSE(3, 2).to(device)  # DenseUNet_sSE
    # model = Unet_gate(3, 2).to(device)
    # model = Unet_global_gate(3, 2).to(device)  #very good
    # model = Unet_sSE1(3, 2).to(device)# 只在编码器部分用sSE
    # model = Unet_cSE1(3, 2).to(device)  # 只在编码器部分用cSE
    # model = Unet_CA1(3, 2).to(device)  # 只在编码器部分用CA
    # model = DenseUnet_cSE_globalgate(3, 2).to(device)  # 只在编码器部分用cSE
    # model = DenseUnet_globalgate(3, 2).to(device)
    # model = DenseUnet_cSE(3, 2).to(device)
    # model = DenseUnet_cSE_nomiddle(3, 2).to(device)
    # model = Unet_sSE1_gate(3, 2).to(device)  #sSE+gate
    # model = FulDenseUnet(3, 2).to(device)
    # model = UnetcSE1gate(3, 2).to(device)
    # model = UnetcSE1gate1(3, 2).to(device)
    # model = Unet_global_gate1(3, 2).to(device)
    # model = Unet_cSE_sSE(3, 2).to(device)
    # model = Unet_BAM(3, 2).to(device)
    # model = Unet_CBAM(3, 2).to(device)
    # model = Unet_global_conv(3, 2).to(device)
    # model = Unet_mul_input(3, 2).to(device)
    # model = Unet_CA_New(3, 2).to(device)
    # model = Unet_CANew_BAM(3, 2).to(device)
    # model = Unet_DAC(3, 2).to(device)
    # model = CE_Net_().to(device)
    # model = Unet_SGE(3,2).to(device)
    # model = se_CE_Net_().to(device)
    # model = Unet_GC(3,2).to(device)
    # model = Unet_GC1(3,2).to(device)
    model = Unet_ResNet34().to(device)
    # # model = Unet_GC_DenseASPP(3,2).to(device)
    # # model = Unet_GC_ResNet34(3,2).to(device)
    # # model = Unet_Mul_ResNet34(3,2).to(device)
    # model = DeepDisc().to(device)
    # # model = Unet_ResNet34_gate().to(device)
    # model = Unet_ResNet34_cSE().to(device)
    # model = Unet_ResNet34_cSE_deep().to(device)
    model = Unet_ResNet34_cSE_nopool_deep().to(device)

    # model = Unet_ResNet34_cSE_deep_aspp().to(device)
    # model = Unet_ResNet34_cSE_deep_gram().to(device)
    # model = Unet_ResNet34_ECA().to(device)
    # model = Unet_ResNet34_gate().to(device)
    # model = Unet_ResNet34_1X1conv().to(device)
    # model = Unet_ResNet34_cSE_down16().to(device)
    # model = Unet_ResNet34_RAU().to(device)
    # model = Unet_ResNet34_cSE_refine().to(device)
    # model = Unet_ResNet34_cSE_aspp().to(device)
    # model = Unet_ResNet34_csSE().to(device)
    # model = Unet_ResNet34_cSE_Deep().to(device)
    # model = Unet_ResNet34_cSE_last().to(device)
    # # model = Unet_ResNet34_GC().to(device)
    # # model = Unet_ResNet34_cSE_spp().to(device)
    # # model = Unet_ResNet34_cSE_skip().to(device)
    # model = Unet_ResNet34_cSE_5().to(device)
    # model = Unet_ResNet34_cSE_full().to(device)
    # model = FCDenseNet103(2).to(device)  # FC-DenseNet103
    # model = FCN().to(device)
    # model = Unet_ResNet34_cSE_decoder().to(device)
    # model = Unet_ResNet34_cSE_CA().to(device)
    # model = Unet_ResNet50().to(device)
    # model = Unet_ResNet18().to(device)
    # model = Unet_ResNet18_cSE().to(device)
    # model = DeepLabV3Plus(n_classes=2,
    #                       n_blocks=[3, 4, 23, 3],
    #                       atrous_rates=[6, 12, 18],
    #                       multi_grids=[1, 2, 4],
    #                       output_stride=16, ).to(device)
    # model = Unet_ResNet34().to(device)

    # no_grad = {
    #     'firstconv.weight',
    #     'firstbn.weight',
    #     'firstbn.bias',
    #     'firstbn.weight',
    #     'encoder1.0.conv1.weight',
    #     'encoder1.0.bn1.weight',
    #     'encoder1.0.bn1.bias',
    #     'encoder1.0.conv2.weight',
    #     'encoder1.0.bn2.weight',
    #     'encoder1.0.bn2.bias',
    #     'encoder1.1.conv1.weight',
    #     'encoder1.1.bn1.weight',
    #     'encoder1.1.bn1.bias',
    #     'encoder1.1.conv2.weight',
    #     'encoder1.1.bn2.weight',
    #     'encoder1.1.bn2.bias',
    #     'encoder1.2.conv1.weight',
    #     'encoder1.2.bn1.weight',
    #     'encoder1.2.bn1.bias',
    #     'encoder1.2.conv2.weight',
    #     'encoder1.2.bn2.weight',
    #     'encoder1.2.bn2.bias',
    #     'encoder2.0.conv1.weight',
    #     'encoder2.0.bn1.weight',
    #     'encoder2.0.bn1.bias',
    #     'encoder2.0.conv2.weight',
    #     'encoder2.0.bn2.weight',
    #     'encoder2.0.bn2.bias',
    #     'encoder2.0.downsample.0.weight',
    #     'encoder2.0.downsample.1.weight',
    #     'encoder2.0.downsample.1.bias',
    #     'encoder2.1.conv1.weight',
    #     'encoder2.1.bn1.weight',
    #     'encoder2.1.bn1.bias',
    #     'encoder2.1.conv2.weight',
    #     'encoder2.1.bn2.weight',
    #     'encoder2.1.bn2.bias',
    #     'encoder2.2.conv1.weight',
    #     'encoder2.2.bn1.weight',
    #     'encoder2.2.bn1.bias',
    #     'encoder2.2.conv2.weight',
    #     'encoder2.2.bn2.weight',
    #     'encoder2.2.bn2.bias',
    #     'encoder2.3.conv1.weight',
    #     'encoder2.3.bn1.weight',
    #     'encoder2.3.bn1.bias',
    #     'encoder2.3.conv2.weight',
    #     'encoder2.3.bn2.weight',
    #     'encoder2.3.bn2.bias',
    #     'encoder3.0.conv1.weight',
    #     'encoder3.0.bn1.weight',
    #     'encoder3.0.bn1.bias',
    #     'encoder3.0.conv2.weight',
    #     'encoder3.0.bn2.weight',
    #     'encoder3.0.bn2.bias',
    #     'encoder3.0.downsample.0.weight',
    #     'encoder3.0.downsample.1.weight',
    #     'encoder3.0.downsample.1.bias',
    #     'encoder3.1.conv1.weight',
    #     'encoder3.1.bn1.weight',
    #     'encoder3.1.bn1.bias',
    #     'encoder3.1.conv2.weight',
    #     'encoder3.1.bn2.weight',
    #     'encoder3.1.bn2.bias',
    #     'encoder3.2.conv1.weight',
    #     'encoder3.2.bn1.weight',
    #     'encoder3.2.bn1.bias',
    #     'encoder3.2.conv2.weight',
    #     'encoder3.2.bn2.weight',
    #     'encoder3.2.bn2.bias',
    #     'encoder3.3.conv1.weight',
    #     'encoder3.3.bn1.weight',
    #     'encoder3.3.bn1.bias',
    #     'encoder3.3.conv2.weight',
    #     'encoder3.3.bn2.weight',
    #     'encoder3.3.bn2.bias',
    #     'encoder3.4.conv1.weight',
    #     'encoder3.4.bn1.weight',
    #     'encoder3.4.bn1.bias',
    #     'encoder3.4.conv2.weight',
    #     'encoder3.4.bn2.weight',
    #     'encoder3.4.bn2.bias',
    #     'encoder3.5.conv1.weight',
    #     'encoder3.5.bn1.weight',
    #     'encoder3.5.bn1.bias',
    #     'encoder3.5.conv2.weight',
    #     'encoder3.5.bn2.weight',
    #     'encoder3.5.bn2.bias',
    #     'encoder4.0.conv1.weight',
    #     'encoder4.0.bn1.weight',
    #     'encoder4.0.bn1.bias',
    #     'encoder4.0.conv2.weight',
    #     'encoder4.0.bn2.weight',
    #     'encoder4.0.bn2.bias',
    #     'encoder4.0.downsample.0.weight',
    #     'encoder4.0.downsample.1.weight',
    #     'encoder4.0.downsample.1.bias',
    #     'encoder4.1.conv1.weight',
    #     'encoder4.1.bn1.weight',
    #     'encoder4.1.bn1.bias',
    #     'encoder4.1.conv2.weight',
    #     'encoder4.1.bn2.weight',
    #     'encoder4.1.bn2.bias',
    #     'encoder4.2.conv1.weight',
    #     'encoder4.2.bn1.weight',
    #     'encoder4.2.bn1.bias',
    #     'encoder4.2.conv2.weight',
    #     'encoder4.2.bn2.weight',
    #     'encoder4.2.bn2.bias',
    # }

    # for name, value in model.named_parameters():
    #     if name in no_grad:
    #         value.requires_grad = False
    #     else:
    #         value.requires_grad = True

    # for name, Value in model.named_parameters():
    #     print('name:{0},\t grad:{1}'.format(name, Value.requires_grad))
    #
    # encoder = []
    # other = []
    #
    # for name, params in model.named_parameters():
    #     if 'encoder' in name:
    #         encoder += [params]
    #     elif 'first' in name:
    #         encoder += [params]
    #     else:
    #         other += [params]

    batch_size = args.batch_size
    # datachoose = args.action
    # criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.BCELoss()  #网络输出一个通道时用这个
    # criterion = torch.nn.CrossEntropyLoss()  # 网络输出多个通道时用这个
    # criterion = MulticlassDiceLoss_CrossEntropyLoss()
    criterion = MulticlassDiceLoss()

    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    # optimizer = torch.optim.SGD(
    #     filter(lambda p: p.requires_grad, model.parameters()),  # 重要的是这一句
    #     lr=0.001)

    # optimizer = optim.Adam(
    #     [
    #         {'params': encoder, 'lr': 0.0001},
    #         {'params': other, 'lr': 0.001},
    #     ],
    # )

    liver_dataset = LiverDataset("./data/DRIVE/", transform=x_transforms, target_transform=y_transforms)
    liver_dataset_val = Val_Dataset("./data/DRIVE/", transform=x_transforms, target_transform=y_transforms)

    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloaders_val = DataLoader(liver_dataset_val, batch_size=1, shuffle=False)

    train_model(model, criterion, optimizer, dataloaders, dataloaders_val)


def test():
    model = Unet(3, 2).to(device).eval()
    # model = Unet_1(3, 2).to(device)  # sSE
    # model = Unet_2(3, 2).to(device)  # cSE
    # model = Unet_CA(3, 2).to(device)  # CA
    # model = FCDenseNet56(2).to(device)  # FC-DenseNet56
    # model = FCDenseNet67(2).to(device)  # FC-DenseNet67
    # model = FCDenseNet103(1).to(device)  # FC-DenseNet103
    # model = Unet_spp(3, 2).to(device)  # Unet_spp
    # model = Unet_CA_spp(3, 2).to(device)  # Unet_CA_spp
    # model = DenseUnet(3, 2).to(device)  # DenseUNet
    # model = Unet_cSE1(3, 2).to(device)  # 只在编码器部分用cSE
    # model = Unet_BAM(3, 2).to(device)
    # model = Unet_gate(3, 2).to(device)
    # model = Unet_CA_New(3, 2).to(device)
    # model = Unet_cSE1(3, 2).to(device)  # 只在编码器部分用cSE
    # model = Unet_GC_ResNet34(3, 2).to(device)
    # model = CE_Net_().to(device)
    # model = Unet_ResNet34().to(device)
    # model = DeepDisc().to(device)
    # model = Unet_ResNet34_cSE().to(device)
    # model = Unet_ResNet34_cSE_down16().to(device)
    # model = FCN().to(device)
    # model = Unet_ResNet18_cSE().to(device)
    # model = Unet_ResNet34_cSE_deep().to(device)
    model = Unet_ResNet34_cSE_nopool_deep().to(device).eval()
    # model = DeepLabV3Plus(n_classes=2,
    #                       n_blocks=[3, 4, 23, 3],
    #                       atrous_rates=[6, 12, 18],
    #                       multi_grids=[1, 2, 4],
    #                       output_stride=16, ).to(device).eval()
    # model = Unet_ResNet34().to(device).eval()

    # model.load_state_dict(torch.load(args.ckp, map_location='cpu'))
    model.load_state_dict(torch.load(args.ckp, map_location={'cuda:1': 'cuda:0'}))
    liver_dataset_val = Val_Dataset("./data/DRIVE/", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset_val, batch_size=1, shuffle=False)

    j = 1
    i = 1
    Dice = 0
    best_Dice = 0
    num_classes = 2
    val_confusion = np.zeros((num_classes, 3))

    with torch.no_grad():  # 不用求导
        # start = time.time()
        start = time.asctime()
        print(start)

        for x, y in dataloaders:
            start = time.asctime()
            print('**** val ****', j)
            # print(j)
            j += 1
            val_inputs = x.to(device)
            val_labels = y.to(device)
            # outputs_val = model(val_inputs)
            ave_out, side_6, side_7, side_8, outputs_val = model(val_inputs)  # 深度监督
            val_preds = torch.max(outputs_val, 1)[1]  # [B,H,W] 元素大小不超过<=2
            val_preds = torch.squeeze(val_preds).cpu().numpy()

            # 极坐标转换
            val_labels = torch.squeeze(val_labels).cpu().numpy()
            val_preds = cv2.linearPolar(val_preds, (112, 112), 112, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
            print('val_preds shape:', val_preds.shape)  # [224,224]
            val_labels = cv2.linearPolar(val_labels, (112, 112), 112, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
            val_labels = np.reshape(val_labels, (1, 1, 224, 224))
            val_labels = torch.tensor(val_labels).to(device)

            val_preds = BW_img(val_preds)

            # 生成图片时不用
            # val_preds = val_preds.reshape(1, 1, val_preds.shape[0], val_preds.shape[1])

        #     for i in range(num_classes):
        #         val_labels_mask = val_labels == i
        #         val_preds_mask = val_preds == i
        #         val_labels = val_labels.long()
        #         TP = np.sum((val_preds == val_labels)[val_labels_mask].data.cpu().numpy())
        #
        #         val_confusion[i, 0] += TP
        #         val_confusion[i, 1] += np.sum((val_labels == val_labels)[val_labels_mask].data.cpu().numpy()) - TP
        #         # val_confusion[i, 2] += np.sum((val_preds == val_preds)[val_preds_mask].data.cpu().numpy()) - TP
        #         val_confusion[i, 2] += np.sum((val_preds == val_preds)[val_preds_mask]) - TP
        # for i in range(1, num_classes):
        #     TP, FP, FN = val_confusion[i]
        #     # print(TP + FP, FN)
        #     dice = (2 * TP) / (FN + FP + 2 * TP)
        #     Jaccard = TP / (FN + TP + FP)
        #     pre = TP / (TP + FP)
        #     recall = TP / (TP + FN)
        #     print('dice is %0.4f,Jaccard is %0.4f,pre is %0.4f,recall is %0.4f' % (dice, Jaccard, pre, recall))

            # 作图
            # val_preds = torch.squeeze(val_preds).cpu().numpy()
            print('shape:', val_preds.shape)
            misc.imsave("result/%d.png" % i, val_preds)
            # imageio.imwrite("result/%d.png" % i, val_preds)
            i += 1
    # end = time.time()
    end = time.asctime()
    print(end)


# def test():
#     model = Unet(3, 1).to(device)
#     # model = Unet_1(3, 1).to(device)  #sSE
#     # model = Unet_2(3, 1).to(device)  #cSE
#     # model = Unet_12(3, 1).to(device)
#     # model = Unet_CA(3, 1).to(device)  # CA
#     # model = FCDenseNet56(1).to(device)  # FC-DenseNet56
#     # model = FCDenseNet67(1).to(device)  # FC-DenseNet67
#     # model = FCDenseNet103(1).to(device)  # FC-DenseNet103
#     # datachoose = args.action
#     model.load_state_dict(torch.load(args.ckp, map_location='cpu'))
#     # liver_dataset = LiverDataset("data/val", datachoose)
#     liver_dataset_val = Val_Dataset("./data/DRIVE/", transform=x_transforms, target_transform=y_transforms)
#     dataloaders = DataLoader(liver_dataset_val, batch_size=1)
#
#     i = 0
#     model.eval()
#     import matplotlib.pyplot as plt
#     plt.ion()
#     with torch.no_grad():  # 不用求导
#         for x, _ in dataloaders:
#             i += 1
#             y = model(x)
#             img_y = torch.squeeze(y).numpy()
#             print(type(img_y))
#             # plt.imshow(img_y)
#             # plt.pause(0.01)
#             cv2.imshow('11', img_y)
#             cv2.waitKey(0)
#         # plt.show()
#     return


if __name__ == '__main__':
    setup_seed(1)
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action == "train":
        train()
    elif args.action == "test":
        test()

    # CUDA_VISIBLE_DEVICES=1 python my_script.py
