# coding: UTF-8
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer
# from pytorch_pretrained import BertModel, BertTokenizer

class Model(nn.Module):

    def __init__(self, bert_file):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_file)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(768, 2)
        # self.load_state_dict(torch.load('bert.ckpt'))

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out
