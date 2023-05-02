#!/usr/bin/env python3
# -*- coding:UTF8 -*-
# ------------------
# @User: ZhangWenhao
# @Mail: psymhmch@163.com
#
# @Based on: ZixiaoChen
# @Mail: 20s151161@stu.hit.edu.cn

# ------------------
import numpy as np
import random
from nltk.corpus import wordnet as wn
from itertools import chain
import spacy
import jieba
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
# 加载英文语言模型
# nlp = spacy.load('en_core_web_sm')
# 加载中文语言模型
nlp = spacy.load('zh_core_web_sm')




replace_dic = {
               'lx':['bj', 'dz', 'mg', 'rb', 'sw', 'zg'],
               'bj':['lx', 'dz', 'mg', 'rb', 'sw', 'zg'],
               'dz':['bj', 'lx', 'mg', 'rb', 'sw', 'zg'],
               'mg':['bj', 'dz', 'lx', 'rb', 'sw', 'zg'],
               'rb':['bj', 'dz', 'mg', 'lx', 'sw', 'zg'],
               'sw':['bj', 'dz', 'mg', 'rb', 'lx', 'zg'],
               'zg':['bj', 'dz', 'mg', 'rb', 'sw', 'lx'],
               'Ma-Weibo_train':['Ma-Weibo_train'],
               'Weibo20_train':['Weibo20_test'],
               'Twitter15_train':['Twitter15_test'],
               'Twitter16_train':['Twitter16_test']

}


def load_seed_words(dataname, percent=1):
    path = './seed_words/'+dataname+'.seed'
    print(path)
    seed_words = {}
    fin = open(path)
    for line in fin:
        line = line.strip()
        if not line:
            continue
        word, weight = line.split()
        if float(weight) <= 0:
            continue
        seed_words[word] = float(weight)
    fin.close()
    save_len = int(len(seed_words) * percent)
    seed_words = sorted(seed_words.items(), key=lambda a: -a[1])
    word_dic = {}
    for word, weight in seed_words:
        word_dic[word] = weight
        save_len -= 1
    return word_dic

def tokenize(text):
    """
    将text分词，并去掉停用词。STOPWORDS -是指Stone, Denis, Kwantes(2010)的stopwords集合.
    :param text:需要处理的文本
    :return:去掉停用词后的"词"序列
    """
    stopwords = open('./stopwords/stopwords.txt', 'r', encoding='utf8').read().split('\n')
    result = [token for token in jieba.lcut(text) if token not in stopwords]
    return result

mask_dir = './augment_data/mask/'
replace_dir = './augment_data/replace/'
sentence_dir = './augment_data/sentence/'

def process(filename):
    method = 'LDA'
    fin = open(filename, 'r', encoding='utf-8', errors='ignore')
    fout_mask = open(mask_dir+filename.split('/')[-1], 'w', encoding='utf-8')
    fout_replace = open(replace_dir+filename.split('/')[-1], 'w', encoding='utf-8')
    fout_sentence = open(sentence_dir+method+'_'+filename.split('/')[-1], 'w', encoding='utf-8')
    dataname = filename.split('/')[-1].split('.raw')[0].strip()
    lines = fin.readlines()
    fin.close()

    fin2 = open(filename, 'r', encoding='utf-8', errors='ignore')
    lines2 = fin2.readlines()
    fin2.close()
    replace_dict={}
    documents=[]
    for i in range(0, len(lines2), 3):
        documents.append(lines2[i])
    processed_docs = [tokenize(doc) for doc in documents]
    word_count_dict = Dictionary(processed_docs)
    # word_count_dict.filter_extremes(no_below=20,no_above=0.1)
    # 删除出现少于6个文档的单词或在30%以上文档中出现的单词
    word_count_dict.filter_extremes(no_below=3, no_above=0.99) # 3 0.1 / 20 0.99
    bag_of_words_corpus = [word_count_dict.doc2bow(pdoc) for pdoc in processed_docs]
    model_name =filename+".lda"#"./ldamodel/" +
    
    # 设置主题数  passes：控制我们在整个语料库上训练模型的频次
    lda_model = gensim.models.LdaModel(bag_of_words_corpus, num_topics=50, id2word=word_count_dict, passes=20)#50 20

    # Ma-Weibo、Weibo20、Twitter15、Twitter16 : no_below=20, no_above=0.1 num_topics=30 passes=20
    # lx、bj、dz、mg、rb、sw、zg : no_below=3或20, no_above=0.1或0.99 num_topics=30或50 passes=20

    lda_model.save(model_name)
    top_topics = lda_model.top_topics(bag_of_words_corpus)
    lda_words=[]
    # ：返回15个 TF/IDF 权重最大的关键词
    topk = 15
    f_look=open(filename+"_topic_words_top20", 'w', encoding='utf-8', errors='ignore')
    for x in top_topics:
        for i in range(topk):
            if i <len(x[0]):
                lda_words.append(x[0][i][1])
        tmp=[]
        for y in x[0][:]:
            tmp.append(y[1])
        # print(f'{filename}：\n', tmp)
        f_look.write(str(tmp)+'\n')
    print()
    print(f'{filename}：\n', lda_words,'\n')
    for i in range(0, len(lines), 3):
        text = lines[i].lower().strip()
        target = lines[i+1].lower().strip()
        stance = lines[i+2].lower().strip() 
        # deriving masked data
        mask_string = text + '\n' + '[MASK]' + '\n' + stance + '\n'
        # deriving replaced data
        random_id = random.randint(0,len(replace_dic[dataname])-1)
        replace_target = replace_dic[dataname][random_id]
        replace_string = text + '\n' + replace_target + '\n' + stance + '\n'
        # deriving masked sentence's data
        text_list = jieba.lcut(text)
        # text_list = text.split()
        sentence = []
        if method=='lda':
            for token in text_list:
                if token in lda_words:
                    sentence.append('[MASK]')
                else:
                    sentence.append(token)
        sentence_string = ''.join(sentence) + '\n' + target + '\n' + stance + '\n'
        # saving data
        fout_mask.write(mask_string)
        fout_replace.write(replace_string)
        fout_sentence.write(sentence_string)
    fout_mask.close()
    fout_replace.close()
    fout_sentence.close()

    

if __name__=="__main__":
    process('./raw_data/lx.raw')
    # process('./raw_data/bj.raw')
    # process('./raw_data/dz.raw')
    # process('./raw_data/mg.raw')
    # process('./raw_data/rb.raw')
    # process('./raw_data/sw.raw')
    # process('./raw_data/zg.raw')
    # process('./raw_data/Ma-Weibo_train.raw')
    # process('./raw_data/Weibo20_train.raw')
    # process('./raw_data/Twitter15_train.raw')
    # process('./raw_data/Twitter16_train.raw')
    

