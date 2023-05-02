# -*- coding: utf-8 -*-
# ------------------

# @Author: ZhangWenhao
# @Mail: psymhmch@outlook.com
#
# @Based on: huwenxing
# @link: https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch
# ------------------
import time
import torch
import numpy as np
from train_eval import train
from model import bert
from utils import build_dataset, build_iterator, get_time_dif


def main(dataset):
    # overtrain_Ma-Weibo overtrain_Weibo20 overtrain_Twitter15 overtrain_Twitter16
    dataset = f'overtrain_{dataset}'
    config = bert.Config(dataset)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = bert.Model(config).to(config.device)
    test_acc, test_report = train(config, model, train_iter, dev_iter, test_iter)
    # 保存实验结果
    tmp = open(f'./{dataset}_10_average_score.txt', 'w', encoding='utf8')
    test_best_result = '10 times average\n>'+test_report
    tmp.write(test_best_result)
    tmp.close()

if __name__ == '__main__':
    datasets = ['Ma-Weibo', 'Weibo20', 'Twitter15', 'Twitter16', 'Zeo-Weibo']
    for dataset in datasets:
        main(dataset)