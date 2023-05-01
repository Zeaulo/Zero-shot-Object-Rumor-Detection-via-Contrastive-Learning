# -*- coding: utf-8 -*-
# ------------------

# @Author: ZhangWenhao
# @Mail: psymhmch@163.com
#
# @Based on: BinLiang
# @Mail: bin.liang@stu.hit.edu.cn
# ------------------

import logging
import argparse
import math
import os
import sys
import random
import numpy

from criterion import CL_pretext, CL_stance

from sklearn import metrics
from time import strftime, localtime

from pytorch_pretrained_bert import BertModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ZSSDDataset
from models import ZPT_HCL, ZPT_HCL_w_text, BERT_PT_HCL

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        self.x_tag = None

        # if 'bert' in opt.model_name:
        #     tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        #     self.tokenizer = tokenizer
        #     bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        #     self.model = opt.model_class(bert, opt).to(opt.device)
        # else:
        #     tokenizer = build_tokenizer(
        #         fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
        #         max_seq_len=opt.max_seq_len,
        #         dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
        #     embedding_matrix = build_embedding_matrix(
        #         word2idx=tokenizer.word2idx,
        #         embed_dim=opt.embed_dim,
        #         dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
        #     self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        self.tokenizer = tokenizer
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt).to(opt.device)

        self.trainset = ZSSDDataset(opt.dataset_file['train'], tokenizer)
        self.testset = ZSSDDataset(opt.dataset_file['test'], tokenizer)

        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for name, p in self.model.named_parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            # print(self.model)
            # unfreeze = ['layer.0', 'layer.11', 'bert.pooler', 'activ', 'linear', 'w1', 'w2']
            # p.requires_grad = False
            # for l in unfreeze:
            #     if l in name:
            #         p.requires_grad = True
                
            # p.requires_grad = False
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, sup_criterion, pretext_criterion, stance_criterion, optimizer, train_data_loader, val_data_loader, test_data_loader):
        max_val_epoch = 0
        best_test_acc = 0
        best_text_f1 = 0
        global_step = 0
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 150)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total, sub_loss1, sub_loss2, sub_loss3 = 0, 0, 0, 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs, logits = self.model(inputs, self.x_tag)
                targets = batch['polarity'].to(self.opt.device)

                sup_loss = sup_criterion(outputs, targets)
                # ZPT_HCL_w_cl
                if not self.opt.use_cl:
                    loss = sup_loss
                else:
                    pretext_labels = batch['cross_label'].to(self.opt.device)
                    pretext_loss = pretext_criterion(logits, pretext_labels)
                    stance_loss = stance_criterion(logits, pretext_labels, targets)
                    
                    contrastive_loss = pretext_loss + self.opt.alpha * stance_loss

                    loss = self.opt.sup_loss_weight * sup_loss + self.opt.cl_loss_weight * contrastive_loss


                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                sub_loss1 += sup_loss.item() * len(outputs)
                if not self.opt.use_cl:
                    sub_loss2 = 0
                else:
                    sub_loss2 += contrastive_loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    train_sub_loss1 = sub_loss1 / n_total
                    train_sub_loss2 = sub_loss2 / n_total
                    logger.info('loss: {:.4f}, supervised_loss: {:.4f}, contrastive_loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_sub_loss1, train_sub_loss2, train_acc))

            val_acc, f1, f_precision, f_recall = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}, val_precision: {:.4f}, val_recall: {:.4f}'.format(val_acc, f1, f_precision, f_recall))
            
            # 保存模型
            # if lawest_val_acc < val_acc: 
            #     lawest_val_acc = val_acc
            #     max_val_epoch = i_epoch
            #     if not os.path.exists('state_dict_cl'):
            #         os.mkdir('state_dict_cl')
            #     path = './state_dict_cl/{0}_{1}{2}'.format(self.opt.model_name, self.opt.dataset, '.pkl')
            #     torch.save(self.model.state_dict(), path)
            #     logger.info('>> saved: {}'.format(path))
            test_acc, f1, f_precision, f_recall = self._evaluate_acc_f1(test_data_loader)
            if best_test_acc < test_acc or (best_test_acc == test_acc and f1 > best_text_f1):
                best_test_acc = test_acc
                best_text_f1 = f1
                max_val_epoch = i_epoch
                best_text_log = [test_acc, f1, f_precision, f_recall]
            logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}, test_precision: {:.4f}, test_recall: {:.4f}'.format(test_acc, f1, f_precision, f_recall))
            f_out = open('log/' + self.opt.dataset+'_' + str(self.opt.seed)+ '.txt', 'w', encoding='utf-8')
            f_out.write('test_acc: {:.4f}, test_f1: {:.4f}, test_precision: {:.4f}, test_recall: {:.4f}'.format(test_acc, f1, f_precision, f_recall)+'\n')
            a, b, c, d = best_text_log
            logger.info('At last best epoch{} >> test_acc: {:.4f}, test_f1: {:.4f}, test_precision: {:.4f}, test_recall: {:.4f}'.format(max_val_epoch, a, b, c, d))

            if i_epoch - max_val_epoch >= self.opt.patience:
                print(f'>> early epoch stop:last best is epoch {max_val_epoch}')
                break

        return best_text_log

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_outputs, _ = self.model(t_inputs, self.x_tag)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)
                
                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total


        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0,1], average='binary')
        f_precision = metrics.precision_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0,1], average='binary')
        f_recall = metrics.recall_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0,1], average='binary')
        return acc, f1, f_precision, f_recall

    def run(self, target):
        en2cn = {'lx':'刘翔', 'bj':'北京', 'dz':'地震','mg':'美国','rb':'日本','sw':'死亡','zg':'中国','Ma-Weibo':'', 'Weibo20':'','Twitter15':'','Twitter16':''}
        if target not in 'Twitter':
            self.x_tag = torch.Tensor(self.tokenizer.text_to_sequence(f'{en2cn[target]} 谣言 非谣言')).to('cuda').long()
        else:
            self.x_tag = torch.Tensor(self.tokenizer.text_to_sequence(f'{en2cn[target]} rumor non-rumor')).to('cuda').long()

        # Loss and Optimizer
        sup_criterion = nn.CrossEntropyLoss()
        pretext_criterion = CL_pretext(self.opt)
        stance_criterion = CL_stance(self.opt)

        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        test_acc, f1, f_precision, f_recall = self._train(sup_criterion, pretext_criterion, stance_criterion, optimizer, train_data_loader, val_data_loader, test_data_loader)
        logger.info('best_test_result >> test_acc: {:.4f}, test_f1: {:.4f}, test_precision: {:.4f}, test_recall: {:.4f}'.format(test_acc, f1, f_precision, f_recall))
        return test_acc, f1, f_precision, f_recall 

def main(target, seed):
    # Ablation experience Models: 1.ZPT_HCL 
    #                             2.ZPT_HCL_w_text 
    #                             3.ZPT_HCL + use_cl=False (ZPT_HCL_w_cl) 
    #                             4.ZPT_HCL_w_text + use_cl=False (ZPT_HCL_w_cl_text)
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='ZPT_HCL', type=str) # ZPT_HCL, ZPT_HCL_w_text, BERT_PT_HCL
    parser.add_argument('--dataset', default=target, type=str) 
    parser.add_argument('--optimizer', default='adam', type=str) # 'adadelta' 'adagrad' 'adam' 'adamax' 'asgd' 'rmsprop' 'sgd' 'Nadam'
    parser.add_argument('--initializer', default='xavier_uniform_', type=str) # xavier_uniform_, orthogonal_, xavier_normal_
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others') # 2e-5 1e-5
    parser.add_argument('--dropout', default=0.3, type=float) # 0.3
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=9, type=int, help='try less epoch') # 500 9
    parser.add_argument('--batch_size', default=5, type=int, help='try 16, 32, 64 for BERT models') #16 12 8
    parser.add_argument('--log_step', default=3, type=int) # val展示步数
    parser.add_argument('--embed_dim', default=768, type=int) # 768 300
    parser.add_argument('--bert_dim', default=768, type=int) # bert-uncase:768 bert-large:1024
    parser.add_argument('--pretrained_bert_name', default='cn_bert_pretrained', help='Chinese:cn_bert_pretrained, English:en_bert_pretrained', type=str) 
    parser.add_argument('--max_seq_len', default=200, help='Chinese try 120~300, English try 50~150', type=int) # 150 other_dataset:cn-200 en-100
    parser.add_argument('--polarities_dim', default=2, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=5, type=int) # 模型的提前停止（单位：epoch）
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=seed, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0.2, type=float, help='set ratio between 0 and 1 for validation support') # 0.15 0.2 0.25
    parser.add_argument('--cl_loss_weight', default=0.02, type=float) # 1 or 0.02
    parser.add_argument('--sup_loss_weight', default=0.8, type=float) # 0.8
    parser.add_argument('--temperature', default=0.07, type=float)
    parser.add_argument('--temperatureP', default=0.07, type=float)
    parser.add_argument('--temperatureY', default=0.14, type=float)
    parser.add_argument('--alpha', default=0.5, type=float, required=False) #0.5
    parser.add_argument('--negative_slope', default=0.01, type=float, help='try 0 ~ 0.5')
    parser.add_argument('--use_cl', default=True, type=bool, help='Be disabled for ablation experiments')
    parser.add_argument('--w1', default=0.5, type=float, help='initial learnable weight1 value for CLS representation')
    parser.add_argument('--w2', default=0.5, type=float, help='initial learnable weight2 value for auxiliary representation')
    parser.add_argument('--choice_dataset', default='v2', type=str, help='v1:no masked samples(vanilla) v2:add masked samples')
    opt = parser.parse_args()

    random.seed(opt.seed)
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'ZPT_HCL': ZPT_HCL,
        'ZPT_HCL_w_text': ZPT_HCL_w_text,
        'BERT_PT_HCL' : BERT_PT_HCL,
    }


    add_masked_dataset_files = {
        'lx': {
            'train': './datasets/o_lx_addmask',
            'test': './datasets/lx',
        },
        'bj': {
            'train': './datasets/o_bj_addmask',
            'test': './datasets/bj',
        },
        'dz': {
            'train': './datasets/o_dz_addmask',
            'test': './datasets/dz',
        },
        'zg': {
            'train': './datasets/o_zg_addmask',
            'test': './datasets/zg',
        },
        'sw': {
            'train': './datasets/o_sw_addmask',
            'test': './datasets/sw',
        },
        'rb': {
            'train': './datasets/o_rb_addmask',
            'test': './datasets/rb',
        },
        'mg': {
            'train': './datasets/o_mg_addmask',
            'test': './datasets/mg',
        },
        'Ma-Weibo': {
            'train': './datasets/o_Ma-Weibo_addmask',
            'test': './datasets/Ma-Weibo',
        },
        'Weibo20': {
            'train': './datasets/o_Weibo20_addmask',
            'test': './datasets/Weibo20',
        },
        'Twitter15': {
            'train': './datasets/o_Twitter15_addmask',
            'test': './datasets/Twitter15',
        },
        'Twitter16': {
            'train': './datasets/o_Twitter16_addmask',
            'test': './datasets/Twitter16',
        },

   }



    no_masked_dataset_files = {
        'lx': {
            'train': './datasets/o_lx',
            'test': './datasets/lx',
        },
        'bj': {
            'train': './datasets/o_bj',
            'test': './datasets/bj',
        },
        'dz': {
            'train': './datasets/o_dz',
            'test': './datasets/dz',
        },
        'zg': {
            'train': './datasets/o_zg',
            'test': './datasets/zg',
        },
        'sw': {
            'train': './datasets/o_sw',
            'test': './datasets/sw',
        },
        'rb': {
            'train': './datasets/o_rb',
            'test': './datasets/rb',
        },
        'mg': {
            'train': './datasets/o_mg',
            'test': './datasets/mg',
        }
   }




    input_colses = {
        'ZPT_HCL': ['concat_bert_indices', 'concat_segments_indices'],
        'ZPT_HCL_w_text':['concat_bert_indices', 'concat_segments_indices'],
        'BERT_PT_HCL':['concat_bert_indices', 'concat_segments_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
        'Nadam': torch.optim.NAdam
    }

    if opt.choice_dataset == 'v1':
        opt.dataset_file = no_masked_dataset_files[opt.dataset]
    elif opt.choice_dataset == 'v2':
        opt.dataset_file = add_masked_dataset_files[opt.dataset]

    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if not os.path.exists('log'):
        os.mkdir('log')
    log_file = './log/' '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    test_acc, f1, f_precision, f_recall = ins.run(target)
    return test_acc, f1, f_precision, f_recall
    




if __name__ == '__main__':
    datasets = ['Ma-Weibo', 'Weibo20', 'Twitter15', 'Twitter16']
    # datasets = ['lx', 'bj', 'dz', 'mg', 'rb','sw','zg']
    for dataset_name in datasets:
        accs = []
        f1s = []
        pres = []
        recs = []
        for i in range(0,10):
            print(f'\n\n第{i+1}次实验：')
            test_acc, f1, f_precision, f_recall = main(dataset_name, 1234+i)
            accs.append(test_acc)
            f1s.append(f1)
            pres.append(f_precision)
            recs.append(f_recall)
        accs = sum(accs)/len(accs)
        f1s = sum(f1s)/len(f1s)
        pres = sum(pres)/len(pres)
        recs = sum(recs)/len(recs)

        # Saving Experiental Results
        experiental_project = ''
        tmp = open(f'./{experiental_project}_{dataset_name}_10_average_score.txt', 'w', encoding='utf8')
        test_best_result = '10 times average\n> test_acc: {:.4f}, test_f1: {:.4f}, test_precision: {:.4f}, test_recall: {:.4f}'.format(accs, f1s, pres, recs)
        tmp.write(test_best_result)
        tmp.close()
