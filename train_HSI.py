# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn

import scipy.io as io

import configs.configs as cfg
import torch.optim as optim

from data.HSICD_data import HSICD_data
from data.get_train_test_set import get_train_test_set as get_set
from tools.train import train as fun_train
from tools.test import test as fun_test
from tools.show import *
from tools.assessment import *

from model.HSImamba import HSI_mamba as fun_model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():

    current_dataset = cfg.current_dataset
    current_model = cfg.current_model
    model_name = current_dataset + current_model
    print('model {}'.format(model_name))
    cfg_data = cfg.data
    cfg_model = cfg.model
    cfg_train = cfg.train['train_model']
    cfg_optim = cfg.train['optimizer']
    cfg_test = cfg.test

    data_sets = get_set(cfg_data)
    img_gt = data_sets['img_gt']
    a = cfg_data['train_data']
    train_data = HSICD_data(data_sets, cfg_data['train_data'])
    test_data = HSICD_data(data_sets, cfg_data['test_data'])
    # Load model
    model = fun_model()
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg_optim['lr'], momentum=cfg_optim['momentum'], weight_decay=cfg_optim['weight_decay'])
    # train
    import time
    a1 = time.perf_counter()
    fun_train(train_data, model, loss_fun, optimizer, device, cfg_train)
    b1 = time.perf_counter()
    # test
    a = time.perf_counter()
    t = False
    pred_train_label, pred_train_acc = fun_test(t, train_data, data_sets['img_gt'], model, device, cfg_test)
    t = True
    pred_test_label, pred_test_acc = fun_test(t, test_data, data_sets['img_gt'], model, device, cfg_test)
    b = time.perf_counter()
    print('训练时间为：',b1-a1)
    print('测试时间为：',b-a)

    # Post processing
    predict_label = torch.cat([pred_train_label, pred_test_label], dim=0)
    print('pred_train_acc {:.2f}%, pred_test_acc {:.2f}%'.format(pred_train_acc, pred_test_acc))
    predict_img = Predict_Label2Img(predict_label, img_gt)

    conf_mat, oa, kappa_co, P, R, F1, acc = accuracy_assessment(img_gt, predict_img)
    assessment_result = [round(oa, 4) * 100, round(kappa_co, 4), round(F1, 4) * 100, round(P, 4) * 100,
                           round(R, 4) * 100, model_name]
    print('assessment_result', assessment_result)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Number of parameters: {total_params / 1e6:.4f}M")

    FP = (predict_img == 1) & (img_gt == 0)
    FN = (predict_img == 0) & (img_gt == 1)
    change = np.zeros(predict_img.shape + (3,), dtype=np.uint8)
    change[FP] = [255, 0, 0]  # 设置 FP 像素为红色
    change[FN] = [0, 0, 255]  # 设置 FN 像素为蓝色
    TP = (predict_img == 1) & (img_gt == 1)
    TN = (predict_img == 0) & (img_gt == 0)
    change[TP] = [255, 255, 255]
    change[TN] = [0, 0, 0]

    # Store
    save_folder = cfg_test['save_folder']
    save_name = cfg_test['save_name']

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    io.savemat(save_folder + '/' + save_name + ".mat",
               {"predict_img": np.array(predict_img.cpu()), "oa": assessment_result})
    imageio.imwrite(save_folder + '/' + save_name + '+predict_img.png', change)
    print('save predict_img successful!')


if __name__ == '__main__':

    main()

