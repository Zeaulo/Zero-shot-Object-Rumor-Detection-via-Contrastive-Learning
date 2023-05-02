# -*- coding: utf-8 -*-
# file: ZPT_HCL_w_text.py
# author: Zhang Wenhao <psymhmch@outlook.com>
# Based on the BERT_PT_HCL.py from binliang <bin.liang@stu.hit.edu.cn>.
import torch
import torch.nn as nn
import torch.nn.functional as F

class ZPT_HCL_w_text(nn.Module):
    def __init__(self, bert, opt):
        super(ZPT_HCL_w_text, self).__init__()
        self.bert = bert
        self.activ = nn.LeakyReLU()     
        self.dropout = nn.Dropout(opt.dropout)
        self.linear = nn.Linear(opt.bert_dim, opt.polarities_dim)
        
    def forward(self, inputs, x_tag=None):
        text_bert_indices, bert_segments_ids = inputs
        # x size - [batch_size, 768]
        _, x = self.bert(text_bert_indices, token_type_ids=bert_segments_ids,output_all_encoded_layers=False) # [batch_size, 200, 768], [batch_size, 768]
        x = torch.relu(x)
        
        x = self.activ(x)
        x = self.linear(x)
        x = self.dropout(x)

        # pooled_output_feature - [batch_size， 768] -> [batch_size, 1, 768] -> [batch_size, 1, 2]
        pooled_output_feature = x.unsqueeze(1)
        pooled_output_feature = F.normalize(pooled_output_feature, dim=2)
        return x, pooled_output_feature
