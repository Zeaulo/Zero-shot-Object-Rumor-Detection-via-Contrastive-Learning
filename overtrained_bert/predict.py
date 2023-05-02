# -*- coding: utf-8 -*-
# ------------------
# @Author: ZhangWenhao
# @Mail: psymhmch@163.com
# ------------------

import torch
from tqdm import tqdm
from pytorch_pretrained import BertTokenizer
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
import os
from model.bert_for_prediction import Model

PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]' 
def build_dataset(file, bert_path, PAD_SIZE):

    def load_dataset(path, pad_size, bert_path):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                
                tokenizer = BertTokenizer.from_pretrained(bert_path)
                token = tokenizer.tokenize(content)
                
                token = [CLS] + token 
                seq_len = len(token)
                mask = []
                token_ids = tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents
    test = load_dataset(f'{os.getcwd()}/overtrain_{file}/data/{file}.txt', PAD_SIZE, bert_path) # text_path, pad_size
    return test

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset):
    iter = DatasetIterater(dataset, 8, torch.device('cuda' if torch.cuda.is_available() else 'cpu') ) # batch_size, device
    return iter



def test(model, test_iter, file):
    # test
    model_dir = f'{os.getcwd()}/overtrain_{file}/saved_dict/bert.ckpt'
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate(model, test_iter, file=file)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)


def evaluate(model, data_iter, file):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    class_list = [x.strip() for x in open(f'{os.getcwd()}/overtrain_{file}/data/class.txt').readlines()] 
    report = metrics.classification_report(labels_all, predict_all, target_names=class_list, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    c = ''
    for i in predict_all:
        c += str(i) + '\n'
    prediction = open (f'{file}_prediction.txt','w',encoding='utf-8')
    prediction.write(c)
    prediction.close()
    return acc, loss_total / len(data_iter), report, confusion

def main(dataset):
    if dataset in ['Ma-Weibo', 'Weibo20', 'Zeo-Weibo']:
        PAD_SIZE = 200
        bert_path = '../cn_bert_pretrained'
    elif dataset in ['Twitter15', 'Twitter16']:
        PAD_SIZE = 100
        bert_path = '../en_bert_pretrained'
    model = Model(bert_path).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu') )
    test_data = build_dataset(dataset, bert_path, PAD_SIZE)
    test_iter = build_iterator(test_data)
    test(model, test_iter, dataset)



# 运行
if __name__ == '__main__':
    datasets = ['Twitter15', 'Twitter16', 'Ma-Weibo', 'Weibo20', 'Zeo-Weibo']
    for dataset in datasets:
        main(dataset)