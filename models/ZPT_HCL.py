# -*- coding: utf-8 -*-
# file: ZPT_HCL.py
# author: Zhang Wenhao <psymhmch@outlook.com>
# Based on the BERT_PT_HCL.py from  bin liang <bin.liang@stu.hit.edu.cn>.
import torch
import torch.nn as nn
import torch.nn.functional as F

def freeze(model, param=False):
    if not param:
        model.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = False

class ZPT_HCL(nn.Module):
    def __init__(self, bert, opt):
        super(ZPT_HCL, self).__init__()
        self.bert = bert
        self.activ = nn.LeakyReLU(opt.negative_slope)     
        self.dropout = nn.Dropout(opt.dropout)
        self.linear = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.w1 = nn.parameter.Parameter(torch.zeros(opt.bert_dim) + opt.w1, requires_grad=False) # True
        self.w2 = nn.parameter.Parameter(torch.zeros(opt.bert_dim) + opt.w2, requires_grad=False) # True
        # self.cl_linear = nn.Linear(opt.bert_dim, opt.bert_dim)
    def forward(self, inputs, x_tag):
        text_bert_indices, bert_segments_ids = inputs
        # x size - [batch_size, 768]
        _, x = self.bert(text_bert_indices, token_type_ids=bert_segments_ids,output_all_encoded_layers=False) # [4, 200, 768], [4, 768]
        x = torch.relu(x)
        
        # x_tag size - [200] -> [batch_size, 200] -> [batch_size, 768]
        x_tag = x_tag.repeat(x.size(0), 1) 
        _,x_tag= self.bert(x_tag, token_type_ids=bert_segments_ids,output_all_encoded_layers=False)
        x_tag = torch.relu(x_tag)

        x = self.w1 * x + self.w2 * x_tag
        # print(f'linear trasform: Wx={float(self.w1.data):.4}\tWx_tag={float(self.w2.data):.4}')
        # print(f'linear trasform: Wx={torch.mean(self.w1.data):.4}\tWx_tag={torch.mean(self.w2.data):.4}')
        x = self.activ(x)
        x = self.linear(x)
        x = self.dropout(x)

        # pooled_output_feature - [batch_size, 768] -> [batch_size, 1, 768] -> [batch_size, 1, 2]
        pooled_output_feature = x.unsqueeze(1)
        pooled_output_feature = F.normalize(pooled_output_feature, dim=2)
        return x, pooled_output_feature
    
    # Another way about ZPT_HCL (not used)
    # def forward(self, inputs, x_tag):
    #     text_bert_indices, bert_segments_ids = inputs
    #     # x size - [8, 768]
    #     _, x = self.bert(text_bert_indices, token_type_ids=bert_segments_ids,output_all_encoded_layers=False) # [4, 200, 768], [4, 768]
    #     x = torch.relu(x)
        
    #     # x_tag size - [200] -> [4,200] -> [4, 768]
    #     x_tag = x_tag.repeat(x.size(0),1) 
    #     _,x_tag= self.bert(x_tag, token_type_ids=bert_segments_ids,output_all_encoded_layers=False)
    #     x_tag = torch.relu(x_tag)

    #     x = (x + x_tag) / 2
    #     x = self.activ(x)
    #     cl_x = x
    #     x = self.dropout(x)
    #     x = self.linear(x)

    #     cl_x = self.dropout(cl_x)
    #     cl_x = self.cl_linear(cl_x)
    #     cl_x = torch.relu(cl_x)
    #     cl_x = self.dropout(cl_x)
    #     cl_x = self.cl_linear(cl_x)
    #     cl_x = x.unsqueeze(1)
    #     cl_x = F.normalize(cl_x, dim=2)
    #     return x, cl_x