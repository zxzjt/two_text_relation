#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#-------------------------------------------------------------------------------
'''
@Author   :   zxzjt
@Contact  :   zhujingtao@loonxi.com
@License  :   (C) Copyright 2013-2018, loonxi
@Software :   PyCharm Community Edition
@File     :   two_text_relation_train.py
@Created  :   7/30/18 3:22 PM
@Version  :   0.0.1
@Desc     :   train two text relation model
'''
#-------------------------------------------------------------------------------

"""Code Base
行到水穷处, 坐看云起时
"""

import os
import numpy as np
import pickle

from two_text_relation.global_log import logger_debug, package_path
from two_text_relation.two_textCNN_relation import TwoTextCNNRelation


def read_file(path):
    with open(path, 'r', encoding='utf8') as f:
        df_list = f.readlines()
        df_list = [string.strip() for string in df_list]
        return df_list

def write_file(path, df_list):
    new_list = [x.replace('\n','') for x in df_list]
    new_list = [x.replace('\r', '')+'\n' for x in new_list]
    with open(path, 'w', encoding='utf8') as f:
        f.writelines(new_list)

def generate_std_keyword_sentence(keyword):
    # 产生含关键词的标准询盘句子
    return "for sale: {0}. whatsapp 1200. pp interested".format(keyword)

def prepare_data(data):
    # data parse
    data_split = [text.split('\t') for text in data]
    sent1 = [text_list[0] +" . "+ text_list[1] for text_list in data_split]
    if len(data_split[0])==4:
        keywords = [text_list[2] for text_list in data_split]
        sent2 = [generate_std_keyword_sentence(keyword) for keyword in keywords]
        label = [int(text_list[3]) for text_list in data_split]
        label = np.array(label)
        return sent1, sent2, label, keywords
    else:
        return sent1


if __name__ == "__main__":
    # prepare data
    print("prepare train data...")
    train_file = './two_text_relation/data/train.txt'
    train_data = read_file(train_file)
    sent1, sent2, label, keywords = prepare_data(train_data)
    #---------------------------------train model---------------------------------#
    two_textCNN_relation_file = "./two_text_relation/two_textCNN_relation.pkl"
    if not os.path.exists(two_textCNN_relation_file):
        two_textCNN_relation_params = {
            # base class TwoTextRelation params
            'MAX_NB_WORDS': 200000,
            'MAX_SEQUENCE_LENGTH': 30,
            'EMBEDDING_DIM': 200,
            # TwoTextCNNRelation params
            'rate_drop_dense': 0.25,
            'epochs': 30,
            'batch_size': 256,
            'filter_num': 32,
            'filter_size': (5, 6, 7, 8),
            'min_sim': 0.5,
            'transform_batch': 10,
            'dense1_unit_num': 100,
            'dense2_unit_num': 100,
            'dense3_unit_num': 200,
            'dense4_unit_num': 200,
            'dense5_unit_num': 200,
            'dense6_unit_num': 200
        }
        two_textCNN_relation = TwoTextCNNRelation(two_textCNN_relation_params)
        corpus_file = './two_text_relation/data/corpus.txt'
        if os.path.exists(two_textCNN_relation.tokenizer_path):
            corpus = [""]
        else:
            corpus = read_file(corpus_file)
            corpus = two_textCNN_relation.text_preprocess(corpus)
        print("preprocess two sentences...")
        sent1 = two_textCNN_relation.text_preprocess(sent1)
        sent2 = two_textCNN_relation.text_preprocess(sent2)
        print("fit two_textCNN_relation...")
        two_textCNN_relation.fit(corpus,sent1,sent2,label)
        two_textCNN_relation.model=None
        print("pickle two_textCNN_relation...")
        pickle.dump(two_textCNN_relation, open(two_textCNN_relation_file, "wb"))
    # ---------------------------------model predict---------------------------------#
    ## load two_textCNN_relation.pkl
    print("load two_textCNN_relation object...")
    two_textCNN_relation1 = pickle.load(open(two_textCNN_relation_file, 'rb'))
    two_textCNN_relation1.load_model()
    print("read test_corpus.txt...")
    test_texts = read_file('./two_text_relation/data/test_corpus.txt')
    test_sents1 = prepare_data(test_texts)
    ## produce api
    print("start predict test_corpus.txt...")
    max_keyword_num = 4
    test_sents1_pre = two_textCNN_relation1.text_preprocess(test_sents1)
    keywords_index = sorted(list(set([key.strip() for key in keywords])))
    std_keyword_sentence = [generate_std_keyword_sentence(keyword) for keyword in keywords_index]
    std_keyword_sentence = two_textCNN_relation1.text_preprocess(std_keyword_sentence)
    print("\n")
    for i,test_sent1 in enumerate(test_sents1_pre):
        recomm_res = two_textCNN_relation1.transform_one_to_many([test_sent1],std_keyword_sentence)
        recomm_keywords = [keywords_index[i] for (i, p) in recomm_res[0:max_keyword_num]]
        print("post and comment: %s" % test_sents1[i])
        print("enquiry keywords:")
        print(recomm_keywords)
        print("\n")

