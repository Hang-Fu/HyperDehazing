"""
定义的是一个数据处理类，针对高光谱影像(小影像)大影像先转化成.tiff格式进行读取
#输入数据路径，与训练集与测试集比例
#获得数据格式
#根据不同数据格式进入不同数据处理模块
#不同数据处理模块进行不同的数据扩充操作
#数据打乱，基于数据比例创建训练集与测试集文件
最终输出的不是张量格式
"""
import os
from osgeo import gdal
import scipy.io as scio
import numpy as np
import cv2
import torch
from einops import rearrange
import math
import random
from sklearn.preprocessing import MinMaxScaler


class DataSet:
    def __init__(self, arg, spatial=False):

        data_path = arg.data_path
        data_name = arg.data_name
        data_rate = arg.data_rate
        img_size = arg.img_size
        ran_dot = arg.ran_dot
        assert os.path.exists(data_path), "path '{}' does not exist.".format(data_path)

        if ran_dot:
            GT_name = arg.GT_name
            self.GT_dir = os.path.join(data_path, GT_name)
            self.GT_type = os.path.splitext(GT_name)[1]
        else:
            train_name = arg.train_name
            self.train_dir = os.path.join(data_path, train_name)
            test_name = arg.test_name
            self.test_dir = os.path.join(data_path, test_name)

        self.image_dir = os.path.join(data_path, data_name)
        self.im_type = os.path.splitext(data_name)[1]
        self.data_rate = data_rate
        self.image_size = img_size
        self.spatial = spatial
        self.ran_dot = ran_dot

    @staticmethod
    def sp_noise(image, prob):
        """
        添加椒盐噪声
        prob:噪声比例
        """
        output = np.zeros(image.shape, np.float)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output

    @staticmethod
    def gasuss_noise(image, mean=0, var=0.001):
        """
            添加高斯噪声
            mean : 均值
            var : 方差
        """
        image = np.array(image / 255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        return out

    def data_lode(self):  # 基于数据路径加载不同数据格式
        """数据加载"""
        assert self.im_type in [".tif", ".mat"], "imagetype must be in ['.tif','.mat']"
        if self.im_type == ".tif":
            dataset = gdal.Open(self.image_dir)
            im_width = dataset.RasterXSize  # 栅格矩阵的列数
            im_height = dataset.RasterYSize  # 栅格矩阵的行数
            img = dataset.ReadAsArray(0, 0, im_width, im_height).astype(np.float)  # 将数据写成数组，对应栅格矩阵
            img = rearrange(img, 'c h w -> h w c')
            del dataset  # 关闭对象，文件dataset
            if self.ran_dot:
                if self.GT_type == ".tif":
                    dataset = gdal.Open(self.GT_dir)
                    im_width = dataset.RasterXSize  # 栅格矩阵的列数
                    im_height = dataset.RasterYSize  # 栅格矩阵的行数
                    labels = dataset.ReadAsArray(0, 0, im_width, im_height).astype(np.float)  # 将数据写成数组，对应栅格矩阵
                    del dataset  # 关闭对象，文件dataset
                else:
                    labels = cv2.imread(self.GT_dir)
            else:
                dataset = gdal.Open(self.train_dir)
                im_width = dataset.RasterXSize  # 栅格矩阵的列数
                im_height = dataset.RasterYSize  # 栅格矩阵的行数
                tr_labs = dataset.ReadAsArray(0, 0, im_width, im_height).astype(np.float)  # 将数据写成数组，对应栅格矩阵
                del dataset  # 关闭对象，文件dataset
                dataset = gdal.Open(self.test_dir)
                im_width = dataset.RasterXSize  # 栅格矩阵的列数
                im_height = dataset.RasterYSize  # 栅格矩阵的行数
                te_labs = dataset.ReadAsArray(0, 0, im_width, im_height).astype(np.float)  # 将数据写成数组，对应栅格矩阵
                del dataset  # 关闭对象，文件dataset

        else:
            data_dict1 = scio.loadmat(self.image_dir)  # need an r!
            my_array = np.array([1, 1])
            for key in data_dict1.keys():
                if type(data_dict1[key]) == type(my_array):
                    img = data_dict1[key]
            if self.ran_dot:
                data_dict1 = scio.loadmat(self.GT_dir)
                for key in data_dict1.keys():
                    if type(data_dict1[key]) == type(my_array):
                        labels = data_dict1[key]
            else:
                data_dict1 = scio.loadmat(self.train_dir)
                for key in data_dict1.keys():
                    if type(data_dict1[key]) == type(my_array):
                        tr_labs = data_dict1[key]
                data_dict1 = scio.loadmat(self.test_dir)
                for key in data_dict1.keys():
                    if type(data_dict1[key]) == type(my_array):
                        te_labs = data_dict1[key]
        if self.ran_dot:
            return img, labels
        else:
            return img, tr_labs, te_labs

    def data_random(self, img, labels):  # 训练过程中对数据进行增强
        flip_prob = 5
        rad1 = np.random.randint(0, high=11, size=1)
        if rad1 < flip_prob:
            img = rearrange(img, 'b c h w -> b c w h')
            labels = rearrange(labels, 'b h w -> b w h')
        # img = torch.tensor(img)
        # labels = torch.tensor(labels)

        if not self.spatial:  # 随机选取
            img = rearrange(img, 'b c h w -> h w b c')
            h = img.shape[0]
            rad = np.random.randint(0, high=2, size=(h, img.shape[1]))  # 返回随机的整数，位于开区间[low, high)
            img = np.append(img[np.where(rad > 0)], img[np.where(rad == 0)], axis=0)
            img = rearrange(img, '(h w) b c -> b c h w', h=h)
            labels = rearrange(labels, 'b h w -> h w b ')
            labels = np.append(labels[np.where(rad > 0)], labels[np.where(rad == 0)], axis=0)
            labels = rearrange(labels, '(h w) b-> b h w', h=h)

        return img, labels

    # def data_crop(self, img, label, batch):
    #     # 先打乱
    #     rate = self.data_rate
    #     rad = np.random.randint(0, high=2, size=batch)  # 返回随机的整数，位于开区间[low, high)
    #     img = np.append(img[np.where(rad > 0)], img[np.where(rad == 0)], axis=0)
    #     label = np.append(label[np.where(rad > 0)], label[np.where(rad == 0)], axis=0)
    #
    #     # 后分配
    #     train = img[:math.floor(batch * rate)]
    #     train_y = label[:math.floor(batch * rate)]
    #
    #     test = img[math.floor(batch * rate):]
    #     img1 = np.array([np.flip(c1, 1) for c1 in test])
    #     test = np.append(test, img1, axis=0)
    #
    #     test_y = label[math.floor(batch * rate):]
    #     img1 = np.array([np.flip(c1, 1) for c1 in test_y])
    #     test_y = np.append(test_y, img1, axis=0)
    #
    #     val = train[math.ceil(train.shape[0] * rate):]
    #     img1 = np.array([np.flip(c1, 1) for c1 in val])
    #     val = np.append(val, img1, axis=0)
    #
    #     val_y = train_y[math.ceil(train.shape[0] * rate):]
    #     img1 = np.array([np.flip(c1, 1) for c1 in val_y])
    #     val_y = np.append(val_y, img1, axis=0)
    #
    #     return train, val, test, train_y, val_y, test_y

    def data_padding(self, img, label=False):

        img_h = img.shape[0]
        img_w = img.shape[1]
        ph1 = self.image_size[0]
        ph2 = self.image_size[1]
        if not (img_h % ph1 and img_w % ph2):
            img = img
        else:
            if img_h % ph1 and not img_w % ph2:
                padding = ph2 - img_w % ph2
                # 在右边扩充
                img = np.append(img, img[:, :padding], axis=0)
            elif not img_h % ph1 and img_w % ph2:
                padding = ph1 - img_h % ph1
                # 在下边扩充
                img = np.append(img, img[:padding, :], axis=1)
            else:
                padding1 = ph1 - img_h % ph1
                padding2 = ph2 - img_w % ph2
                # 在右，下边扩充
                img = np.append(img, img[:, :padding2], axis=1)
                img = np.append(img, img[:padding1, :], axis=0)
        if label:
            img = rearrange(img, '(h p1) (w p2) -> (h w) p1 p2', p1=ph1, p2=ph2)
        else:
            hsi = img
            hsi_1 = hsi
            hsi = hsi.reshape(hsi_1.shape[0] * hsi_1.shape[1], hsi_1.shape[2])
            hsi = self.band_wise_normlization(hsi)
            hsi = hsi.reshape(hsi_1.shape[0], hsi_1.shape[1], hsi_1.shape[2])
            img = hsi
            img = rearrange(img, '(h p1) (w p2) c -> (h w) p1 p2 c', p1=ph1, p2=ph2)
        return img

    def data_tribute1(self, img, label, TR=False):  # 配置训练集验证集与测试集,可以在此处添加波段选择模块
        """训练集，测试集分离样本可以直接用，先裁剪再进行洗牌,训练集可选择进行扩充"""

        hsi = img
        gth = label
        del img
        num_class = int(np.max(label))
        del label
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~normlization~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        hsi_1 = hsi
        hsi = hsi.reshape(hsi_1.shape[0] * hsi_1.shape[1], hsi_1.shape[2])
        hsi = self.band_wise_normlization(hsi)
        hsi = hsi.reshape(hsi_1.shape[0], hsi_1.shape[1], hsi_1.shape[2])
        del hsi_1
        r = self.image_size[0] // 2
        hsi = np.pad(hsi, ((r, r), (r, r), (0, 0)), 'symmetric')
        gth = np.pad(gth, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
        Xh1 = []
        Y1 = []
        # 随机各类寻找规定数目样本
        for c in range(1, num_class + 1):  # 训练集
            idx, idy = np.where(gth == c)
            ID = np.random.permutation(len(idx))
            idx = idx[ID]
            idy = idy[ID]
            if TR:
                for i in range(len(idx)):
                    tmph1 = hsi[idx[i] - r:idx[i] + r+1, idy[i] - r:idy[i] + r+1, :]
                    tmpy1 = gth[idx[i], idy[i]]  # 应不应该-1
                    Xh1.append(tmph1)  # 原图
                    Xh1.append(np.flip(tmph1, axis=0))  # 镜像
                    noise = np.random.normal(0.0, 0.01, size=tmph1.shape)
                    Xh1.append(np.flip(tmph1 + noise, axis=1))  # 加噪声然后随机镜像
                    k = np.random.randint(4)
                    Xh1.append(np.rot90(tmph1, k=k))  # 随机旋转
                    Y1.append(tmpy1)
                    Y1.append(tmpy1)
                    Y1.append(tmpy1)
                    Y1.append(tmpy1)
            else:
                for i in range(len(idx)):
                    tmph1 = hsi[idx[i] - r:idx[i] + r+1, idy[i] - r:idy[i] + r+1, :]
                    tmpy1 = gth[idx[i], idy[i]]  # 应不应该-1
                    Xh1.append(tmph1)  # 原图
                    Y1.append(tmpy1)

        index = np.random.permutation(len(Xh1))
        Xh1 = np.asarray(Xh1, dtype=np.float32)  # 将列表转换为数组
        Y1 = np.asarray(Y1, dtype=np.int8)
        Xh1 = Xh1[index, ...]
        Y1 = Y1[index]
        return Xh1, Y1

    def data_tribute2(self, img, label, append=False, enhance=False):
        """随机撒点空间不相交样本"""
        per = self.data_rate
        hsi = img
        gth = label
        del img
        num_class = int(np.max(label))
        del label
        # r = self.image_size[0] // 2
        # hsi = np.pad(hsi, ((r, r), (r, r), (0, 0)), 'symmetric')
        # gth = np.pad(gth, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~normlization~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        hsi_1 = hsi
        hsi = hsi.reshape(hsi_1.shape[0] * hsi_1.shape[1], hsi_1.shape[2])
        hsi = self.band_wise_normlization(hsi)
        hsi = hsi.reshape(hsi_1.shape[0], hsi_1.shape[1], hsi_1.shape[2])
        del hsi_1

        # 随机各类寻找规定数目样本
        tmph1 = np.zeros(hsi.shape, np.float)
        tmpy1 = np.zeros(gth.shape, np.float)
        tmph2 = np.zeros(hsi.shape, np.float)
        tmpy2 = np.zeros(gth.shape, np.float)
        for c in range(1, num_class + 1):  # 训练集
            idx, idy = np.where(gth == c)
            ID = np.random.permutation(len(idx))
            idx = idx[ID]
            idy = idy[ID]
            if per < 1:
                idx1 = idx[:int(per * len(idx))]
                idy1 = idy[:int(per * len(idy))]
                idx2 = idx[int(per * len(idx)):]
                idy2 = idy[int(per * len(idy)):]
            else:
                idx1 = idx[:int(per)]
                idy1 = idy[:int(per)]
                idx2 = idx[int(per):]
                idy2 = idy[int(per):]

            tmph1[idx1, idy1] = hsi[idx1, idy1]
            tmpy1[idx1, idy1] = gth[idx1, idy1] - 1

            tmph2[idx2, idy2] = hsi[idx2, idy2]
            tmpy2[idx2, idy2] = gth[idx2, idy2] - 1
        # 已经获取训练集或测试集
        # 训练集
        tr, tr_y = self.data_tribute1(tmph1, tmpy1, append, enhance)
        te, te_y = self.data_tribute1(tmph2, tmpy2, enhance)
        return tr, tr_y, te, te_y

    def data_tribute3(self, img, label):
        """分类网络样本集设计,随机撒点"""
        per = self.data_rate
        hsi = img
        gth = label
        del img
        num_class = int(np.max(label))
        del label
        r = self.image_size[0] // 2
        hsi = np.pad(hsi, ((r, r), (r, r), (0, 0)), 'symmetric')
        gth = np.pad(gth, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~normlization~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        hsi_1 = hsi
        hsi = hsi.reshape(hsi_1.shape[0] * hsi_1.shape[1], hsi_1.shape[2])
        hsi = self.band_wise_normlization(hsi)
        hsi = hsi.reshape(hsi_1.shape[0], hsi_1.shape[1], hsi_1.shape[2])
        del hsi_1

        Xh1 = []
        Y1 = []
        Xh2 = []
        Y2 = []
        # 随机各类寻找规定数目样本
        for c in range(1, num_class + 1):  # 训练集
            idx, idy = np.where(gth == c)
            ID = np.random.permutation(len(idx))
            idx = idx[ID]
            idy = idy[ID]
            if per < 1:
                idx1 = idx[:int(per * len(idx))]
                idy1 = idy[:int(per * len(idy))]
                # 测试集
                idx2 = idx[int(per * len(idx)):]
                idy2 = idy[int(per * len(idy)):]
                # idx2 = idx[:int(per + 0.3 * len(idx))]
                # idy2 = idy[:int(per + 0.3 * len(idy))]

            else:
                # if int(per) * 2 > len(idx):
                #     idx1 = idx[:int(0.3 * int(per))]
                #     idy1 = idy[:int(0.3 * int(per))]
                #     # 测试集
                #     idx2 = idx[int(0.3 * int(per)):]
                #     idy2 = idy[int(0.3 * int(per)):]
                # else:
                idx1 = idx[:int(per)]
                idy1 = idy[:int(per)]
                if len(idx) < int(per)+300:
                    idx2 = idx[int(per):]
                    idy2 = idy[int(per):]
                else:
                    idx2 = idx[int(per):int(per) + 300]
                    idy2 = idy[int(per):int(per) + 300]

            for i in range(len(idx1)):
                tmph1 = hsi[idx1[i] - r:idx1[i] + r+1, idy1[i] - r:idy1[i] + r+1, :]
                tmpy1 = gth[idx1[i], idy1[i]] - 1
                Xh1.append(tmph1)  # 原图
                Xh1.append(np.flip(tmph1, axis=0))  # 镜像
                noise = np.random.normal(0.0, 0.01, size=tmph1.shape)
                Xh1.append(np.flip(tmph1 + noise, axis=1))  # 加噪声然后随机镜像
                k = np.random.randint(4)
                Xh1.append(np.rot90(tmph1, k=k))  # 随机旋转
                Y1.append(tmpy1)
                Y1.append(tmpy1)
                Y1.append(tmpy1)
                Y1.append(tmpy1)
            for i in range(len(idx2)):
                tmph2 = hsi[idx2[i] - r:idx2[i] + r+1, idy2[i] - r:idy2[i] + r+1, :]
                tmpy2 = gth[idx2[i], idy2[i]] - 1
                Xh2.append(tmph2)  # 原图
                Y2.append(tmpy2)
        index = np.random.permutation(len(Xh1))
        Xh1 = np.asarray(Xh1, dtype=np.float32)  # 将列表转换为数组
        Y1 = np.asarray(Y1, dtype=np.int8)
        Xh1 = Xh1[index, ...]
        Y1 = Y1[index]
        index = np.random.permutation(len(Xh2))
        Xh2 = np.asarray(Xh2, dtype=np.float32)  # 将列表转换为数组
        Y2 = np.asarray(Y2, dtype=np.int8)
        Xh2 = Xh2[index, ...]
        Y2 = Y2[index]
        return Xh1, Y1, Xh2, Y2

    @staticmethod
    def band_wise_normlization(data_spat, flag='trn'):
        scaler = MinMaxScaler(feature_range=(0, 1))

        spat_data = data_spat.reshape(-1, data_spat.shape[-1])
        data_spat_new = scaler.fit_transform(spat_data).reshape(data_spat.shape)

        # print('{}_spat:{}'.format(flag, data_spat_new.shape))
        # print('{} Spatial dataset normalization Finished!'.format(flag))
        return data_spat_new

    def data_add(self, train, train_y):
        """旋转镜像加噪声"""
        t_y = train_y
        t = train
        if self.image_size[0] == self.image_size[1]:
            img1 = np.array([np.flip(c1, 1) for c1 in train])
            img2 = np.array([np.rot90(c1, -1) for c1 in train])
            img3 = np.array([np.rot90(c1, 1) for c1 in train])
            img4 = np.array([np.rot90(c1, 2) for c1 in train])

            label1 = np.array([np.flip(c1, 1) for c1 in train_y])
            label2 = np.array([np.rot90(c1, -1) for c1 in train_y])
            label3 = np.array([np.rot90(c1, 1) for c1 in train_y])
            label4 = np.array([np.rot90(c1, 2) for c1 in train_y])
            train = np.append(train, img1, axis=0)  # img2, img3, img4,
            train = np.append(train, img2, axis=0)  # img2, img3, img4,
            train = np.append(train, img3, axis=0)  # img2, img3, img4,
            train = np.append(train, img4, axis=0)  # img2, img3, img4,
            train_y = np.append(train_y, label1, axis=0)
            train_y = np.append(train_y, label2, axis=0)
            train_y = np.append(train_y, label3, axis=0)
            train_y = np.append(train_y, label4, axis=0)

        # 添加椒盐噪声，噪声比例为 0.02
        out1 = np.array([self.sp_noise(c1, prob=0.02) for c1 in t])
        # 添加高斯噪声，均值为0，方差为0.001
        # out2 = np.array([self.gasuss_noise(c1, mean=0, var=0.001) for c1 in t])#现在不适合添加高斯噪声
        train = np.append(train, out1, axis=0)
        # train = np.append(train, out2, axis=0)
        train_y = np.append(train_y, t_y, axis=0)
        # train_y = np.append(train_y, t_y, axis=0)
        return train, train_y

    def data_en(self, img, label):
        """波段选择等操作"""
        return img, label

    def end_data(self, img):
        img_w = img.shape[1]
        num_w = math.ceil(img_w / self.image_size[0])
        img = self.data_padding(img)
        img = rearrange(img, 'b h w c -> b c h w')
        return img, num_w

    def data_random1(self, img, labels):  # 训练过程中对数据进行增强
        flip_prob = 5
        rad1 = np.random.randint(0, high=11, size=1)
        if rad1 < flip_prob:
            img = rearrange(img, 'b c h w -> b c w h')

        img = torch.tensor(img)
        labels = torch.tensor(labels)
        return img, labels
