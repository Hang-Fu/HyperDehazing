import os,argparse
import numpy as np
from PIL import Image
from models import *
import torch
import cv2
import torch.nn as nn
import torchvision.transforms as tfs 
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from osgeo import gdal
from util import tensor2im
gdal.PushErrorHandler("CPLQuietErrorHandler")
from metrics import psnr, ssim, UQI, SAM
from models.HyperDehazeNet import HyperDehazeNet


def TwoPercentLinear(image, max_out=255, min_out=0):  # 2%的线性拉伸
    b, g, r = cv2.split(image)  # 分开三个波段

    def gray_process(gray, maxout = max_out, minout = min_out):
        high_value = np.percentile(gray, 98)  # 取得98%直方图处对应灰度
        low_value = np.percentile(gray, 2)    # 同理
        truncated_gray = np.clip(gray, a_min=low_value, a_max=high_value)
        processed_gray = ((truncated_gray - low_value)/(high_value - low_value)) * (maxout - minout)#线性拉伸嘛
        return processed_gray
    r_p = gray_process(r)
    g_p = gray_process(g)
    b_p = gray_process(b)
    result = cv2.merge((b_p, g_p, r_p)) #合并处理后的三个波段
    return np.uint8(result)


def get_write_picture_fina(img):  # get_write_picture函数得到训练过程中的可视化结果
    img = tensor2im(img, np.float)
    img = img.astype(np.uint8)
    output = TwoPercentLinear(img[:, :, (3, 2, 1)])
    return output


abs = os.getcwd()+'/'
parser = argparse.ArgumentParser()
# parser.add_argument('--task', type=str, default='its', help='its or ots')
parser.add_argument('--test_imgs', type=str, default=r'F:\GF-5 dehaze\test\hazy', help='Test imgs folder')
parser.add_argument('--output_dir', type=str, default='./pre_img/')
opt = parser.parse_args()
dataset = opt.task
img_dir = opt.test_imgs+'/'
model_dir = abs+f'trained_models/train_HyperDehazeNet.pk'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckp = torch.load(model_dir, map_location=device)
net = HyperDehazeNet()
net = nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net = net.module.to(torch.device('cpu'))
net.eval()

for im in os.listdir(img_dir):
    print(f'\r {im}', end='', flush=True)
    haze = gdal.Open(img_dir+im).ReadAsArray().astype(np.float32)
    haze = haze / np.max(haze)
    haze = np.expand_dims(haze, 0)
    # print(type(haze), haze.shape)
    haze = torch.from_numpy(haze).type(torch.FloatTensor)
    with torch.no_grad():
        pred = net(haze)
    ts = torch.squeeze(pred.cpu())

    write_image2 = get_write_picture_fina(pred)
    write_image_name2 = opt.output_dir + "/result" + '_' + str(im) + ".tif"  # 待保存的训练可视化结果路径与名称
    Image.fromarray(np.uint8(write_image2)).save(write_image_name2)
