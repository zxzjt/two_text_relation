#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#-------------------------------------------------------------------------------
'''
@Author   :   zxzjt
@Contact  :   zhujingtao@loonxi.com
@License  :   (C) Copyright 2013-2018, loonxi
@Software :   PyCharm Community Edition
@File     :   two_textCNN_relation.py
@Created  :   2018/7/26 20:19
@Version  :   0.0.1
@Desc     :   two_textCNN_relation
'''
#-------------------------------------------------------------------------------

"""Code Base
keras代码比tensorflow简洁10倍
"""

import os
import numpy as np
from gensim import models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,GRU,GlobalAveragePooling1D,Conv1D
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import keras.backend as K
from keras.layers import Lambda

from .global_log import logger_debug,package_path
from .text_preprocessing import TextProcessor

os.environ['CUDA_VISIBLE_DEVICES']='1'
model_path = package_path+"model/"


class TwoTextRelation(object):
    """两文本相似基类
    两文本相似基类

    参数
    MAX_NB_WORDS: 最大词汇量
    MAX_SEQUENCE_LENGTH: 句子最大长度
    EMBEDDING_DIM: 词向量维度

    属性
    MAX_NB_WORDS: 最大词汇量
    MAX_SEQUENCE_LENGTH: 句子最大长度
    EMBEDDING_DIM: 词向量维度

    text_preprocessor: 预处理器
    tokenizer_path: tokenizer路径
    embedding_matrix_path: embedding_matrix路径
    wv_path: wv路径
    tokenizer: 分词器
    embedding_matrix: 词向量矩阵
    """
    def __init__(self, params):
        # 加载预处理模块
        self.text_preprocessor = TextProcessor()
        # params init
        self.MAX_NB_WORDS = params['MAX_NB_WORDS']
        self.MAX_SEQUENCE_LENGTH = params['MAX_SEQUENCE_LENGTH']
        self.EMBEDDING_DIM = params['EMBEDDING_DIM']
        self.tokenizer_path = model_path+'tokenizer.pkl'
        self.embedding_matrix_path = model_path+"embedding_matrix.npy"
        self.wv_path = model_path+'w2v.model'

    def text_preprocess(self, sentences):
        # 预处理文本
        sentences_pre = [self.text_preprocessor.text_process(string, res='string') for string in sentences]
        return sentences_pre

    def fit_tokenizer(self, corpus):
        """fit分词器
        fit分词器
        :param corpus: 语料
        :return: None
        """
        if os.path.exists(self.tokenizer_path):
            print("loading tokenizer...")
            self.tokenizer = pickle.load(open(self.tokenizer_path, 'rb'))
        else:
            print("Fit tokenizer...")
            self.tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS, lower=False)
            self.tokenizer.fit_on_texts(corpus)
            print("Save tokenizer...")
            pickle.dump(self.tokenizer, open(self.tokenizer_path, "wb"))
        print('Found %s unique tokens' % len(self.tokenizer.word_index))

    def sent2seq(self, sentences):
        """把句子转换成序列
        把句子转换成序列，如‘如何 来 防治 水稻 稻瘟病’----->[6, 383, 2, 1, 12]
        :param sentences: sentences list
        :return: 序列 np.array
        """
        sequences = self.tokenizer.texts_to_sequences(sentences)
        sequences = pad_sequences(sequences, maxlen=self.MAX_SEQUENCE_LENGTH)  # 维度统一为MAX_SEQUENCE_LENGTH，不足的补0
        return sequences

    def generate_embedding_matrix(self, corpus, nb_words):
        """generate embedding matrix
        generate embedding matrix
        :param corpus: 语料
        :param nb_words: 词汇量
        :return: None
        """

        if os.path.exists(self.embedding_matrix_path):
            print("loading embedding_matrix...")
            self.embedding_matrix = np.load(self.embedding_matrix_path)
        else:
            if os.path.exists(self.wv_path):
                print("loading wv...")
                wv = models.Word2Vec.load(self.wv_path).wv
            else:
                print("no w2v.model find...")
                # 先启动一个空模型 an empty model
                word2vec_model = models.Word2Vec(size=200, window=7, min_count=3)
                texts_list = [text.split(' ') for text in corpus]
                word2vec_model.build_vocab(texts_list)
                # train word2vec
                print("train word2vec...")
                for epoch in range(50):
                    print("epoch %d ..." % epoch)
                    word2vec_model.train(texts_list, total_examples=word2vec_model.corpus_count,epochs=word2vec_model.iter)
                    # print("loss of epoch %d: %s" % (epoch, word2vec_model.get_latest_training_loss()))
                wv = word2vec_model.wv
                # save model
                print("save w2v.model ...")
                word2vec_model.save(self.wv_path)
            print("wv vocab num %s" % len(wv.vocab))
            print('preparing embedding matrix')
            self.embedding_matrix = np.zeros((nb_words, self.EMBEDDING_DIM))
            for word, i in self.tokenizer.word_index.items():
                if word in wv.vocab.keys():
                    self.embedding_matrix[i] = wv[word]
            print('Null word embeddings: %d' % np.sum(np.sum(self.embedding_matrix, axis=1) == 0))
            wv = None
            print("save embedding_matrix...")
            np.save(self.embedding_matrix_path, self.embedding_matrix)
        print("embedding matrix shape (%s, %s)" % self.embedding_matrix.shape)


class TwoTextCNNRelation(TwoTextRelation):
    """利用CNN的文本相似度计算
    利用CNN的文本相似度计算

    参数
    rate_drop_dense: dense drop rate
    epochs: train epochs
    batch_size: batch size
    filter_num: conv num
    filter_size: conv size
    min_sim: 相似度阈值
    transform_batch: transform batch
    dense1_unit_num: dense1 unit num
    dense2_unit_num: dense2 unit num
    dense3_unit_num: dense3 unit num
    dense4_unit_num: dense4 unit num
    dense5_unit_num: dense5 unit num
    dense6_unit_num: dense6 unit num

    属性
    rate_drop_dense: dense drop rate
    epochs: train epochs
    batch_size: batch size
    filter_num: conv num
    filter_size: conv size
    min_sim: 相似度阈值
    transform_batch: transform batch
    dense1_unit_num: dense1 unit num
    dense2_unit_num: dense2 unit num
    dense3_unit_num: dense3 unit num
    dense4_unit_num: dense4 unit num
    dense5_unit_num: dense5 unit num
    dense6_unit_num: dense6 unit num

    bst_model_path: 模型路径
    model: 模型
    """
    def __init__(self, params):
        # init base
        super(TwoTextCNNRelation, self).__init__(params)
        # params init
        self.rate_drop_dense = params['rate_drop_dense']
        self.epochs = params['epochs']
        self.batch_size= params['batch_size']
        self.filter_num = params['filter_num']
        self.filter_size = params['filter_size']
        self.min_sim = params['min_sim']
        self.transform_batch = params['transform_batch']
        self.dense1_unit_num = params['dense1_unit_num']
        self.dense2_unit_num = params['dense2_unit_num']
        self.dense3_unit_num = params['dense3_unit_num']
        self.dense4_unit_num = params['dense4_unit_num']
        self.dense5_unit_num = params['dense5_unit_num']
        self.dense6_unit_num = params['dense6_unit_num']
        self.bst_model_path = model_path + 'two_textCNN_relation.h5'

    def manhattan_distance(self, pair):
        # manhattan distance
        result = K.exp(-K.sum(K.abs(pair[0] - pair[1]), axis=1, keepdims=True))
        return result

    def abs_distance(self, pair):
        # abs distance
        result = K.abs(pair[0] - pair[1])
        return result

    def construct_model(self, nb_words):
        """
        定义模型结构
        :param nb_words: 词汇量
        :return: None
        """
        print("start to construct two_textCNN_relation.h5...")
        embedding_layer = Embedding(nb_words,
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.MAX_SEQUENCE_LENGTH,
                                    trainable=True)
        conv1 = Conv1D(filters=self.filter_num, kernel_size=self.filter_size[0], padding='same', activation='relu')
        conv2 = Conv1D(filters=self.filter_num, kernel_size=self.filter_size[1], padding='same', activation='relu')
        conv3 = Conv1D(filters=self.filter_num, kernel_size=self.filter_size[2], padding='same', activation='relu')
        conv4 = Conv1D(filters=self.filter_num, kernel_size=self.filter_size[3], padding='same', activation='relu')

        dense_1 = Dense(self.dense1_unit_num, activation='relu')
        dense_2 = Dense(self.dense2_unit_num, activation='relu')
        dense_3 = Dense(self.dense3_unit_num, activation='relu')
        dense_4 = Dense(self.dense4_unit_num, activation='relu')
        dense_5 = Dense(self.dense5_unit_num, activation='relu')
        dense_6 = Dense(self.dense6_unit_num, activation='relu')
        #dense_7 = Dense(200, activation='relu')
        #dense_8 = Dense(200, activation='relu')
        #dense_9 = Dense(200, activation='relu')
        #dense_10 = Dense(200, activation='relu')

        sequence_1_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences_1 = embedding_layer(sequence_1_input)

        x1 = conv1(embedded_sequences_1)
        x1 = GlobalAveragePooling1D()(x1)

        x2 = conv2(embedded_sequences_1)
        x2 = GlobalAveragePooling1D()(x2)

        x3 = conv3(embedded_sequences_1)
        x3 = GlobalAveragePooling1D()(x3)

        x4 = conv4(embedded_sequences_1)
        x4 = GlobalAveragePooling1D()(x4)
        x = concatenate([x1, x2, x3, x4])

        x = Dropout(self.rate_drop_dense)(x)
        x = BatchNormalization()(x)
        x = dense_1(x)

        x = Dropout(self.rate_drop_dense)(x)
        x = BatchNormalization()(x)
        x = dense_2(x)

        sequence_2_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences_2 = embedding_layer(sequence_2_input)

        y1 = conv1(embedded_sequences_2)
        y1 = GlobalAveragePooling1D()(y1)

        y2 = conv2(embedded_sequences_2)
        y2 = GlobalAveragePooling1D()(y2)

        y3 = conv3(embedded_sequences_2)
        y3 = GlobalAveragePooling1D()(y3)

        y4 = conv4(embedded_sequences_2)
        y4 = GlobalAveragePooling1D()(y4)
        y = concatenate([y1, y2, y3, y4])

        y = Dropout(self.rate_drop_dense)(y)
        y = BatchNormalization()(y)
        y = dense_1(y)

        y = Dropout(self.rate_drop_dense)(y)
        y = BatchNormalization()(y)
        y = dense_2(y)

        def abs_distance(pair):
            result = K.abs(pair[0] - pair[1])
            return result

        dist1 = Lambda(abs_distance)([x, y])
        dist2 = dense_3(dist1)
        dist3 = dense_4(dist2)
        dist4 = dense_5(dist3)
        dist = dense_6(dist4)
        # dist6 = dense_7(dist5)
        # dist7 = dense_8(dist6)
        # dist8 = dense_9(dist7)
        # dist = dense_10(dist8)

        preds = Dense(1, activation='sigmoid')(dist)

        self.model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=[preds])

        self.model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
        print("two_textCNN_relation.h5 summary...")
        self.model.summary()

    def train_model(self, seq1, seq2, labels, validation_split = 0.15):
        """训练模型
        训练模型
        :param seq1: 序列1
        :param seq2: 序列2
        :param labels: labels
        :param validation_split: 验证集比例
        :return: None
        """
        print("fit two_textCNN_relation.h5...")
        #early_stopping = EarlyStopping(monitor='val_loss', patience=30)
        #model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)
        cw = {0: 1, 1: 1}
        hist = self.model.fit([seq1, seq2], labels,
                         # validation_data=([test_seq1[:-100], test_seq2[:-100]], test_labels[:-100]),
                         validation_split=validation_split,
                         epochs=self.epochs, batch_size=self.batch_size,
                         shuffle=True, class_weight=cw, callbacks=None)
        # model.load_weights(bst_model_path)
        print("save two_textCNN_relation.h5...")
        self.model.save(self.bst_model_path)
        print("train bst_acc %s, train bst_loss %s" % (max(hist.history['acc']), min(hist.history['loss'])))
        print("val bst_acc %s, val bst_loss %s" % (max(hist.history['val_acc']), min(hist.history['val_loss'])))

    def fit(self, corpus, sentences1, sentences2, labels):
        """fit two_textCNN_relation
        fit two_textCNN_relation
        :param corpus: corpus to fit tokenizer
        :param sentences1: sentences1
        :param sentences2: sentences2
        :param labels: labels
        :return: model
        """
        self.fit_tokenizer(corpus)
        seq1 = self.sent2seq(sentences1)
        seq2 = self.sent2seq(sentences2)
        nb_words = min(self.MAX_NB_WORDS, len(self.tokenizer.word_index)) + 1
        self.generate_embedding_matrix(corpus, nb_words)
        self.construct_model(nb_words)
        self.train_model(seq1, seq2, labels)
        return self.model

    def transform_one_to_many(self, sentence1, sentences2):
        """预测某一句子与一组标准句子集的相似度
        预测某一句子与一组标准句子集的相似度
        :param sentence1: 待预测的新句子
        :param sentences2: 一组标准句子集
        :return: 相似index和相似度, [(i, sim),]
        """
        seq1 = self.sent2seq(sentence1)
        seq1 = seq1.repeat(len(sentences2), axis=0)
        seq2 = self.sent2seq(sentences2)
        predicts = self.model.predict([seq1, seq2], batch_size=self.transform_batch, verbose=0)
        recomm_res = [(i, p[0]) for i,p in enumerate(predicts) if p[0] < 1 - self.min_sim]
        recomm_res = sorted(recomm_res, key=lambda res:res[1], reverse=True)
        return recomm_res

    def transform_many_to_many(self, sentences1, sentences2):
        """预测某一组句子与一组标准句子集的相似度
        预测某一组句子与一组标准句子集的相似度
        :param sentences1: 待预测的新一组句子
        :param sentences2: 一组标准句子集
        :return: 相似index和相似度, [(i, sim),]
        """
        seq1 = self.sent2seq(sentences1)
        seq2 = self.sent2seq(sentences2)
        predicts = self.model.predict([seq1, seq2], batch_size=self.transform_batch, verbose=0)
        recomm_res = [(i, p[0]) for i,p in enumerate(predicts) if p[0] < 1 - self.min_sim]
        return recomm_res

    def evaluate(self, sentences1, sentences2, labels):
        """model evaluate
        model evaluate
        :param sentences1: sentences1
        :param sentences2: sentences2
        :param labels: labels
        :return: score
        """
        seq1 = self.sent2seq(sentences1)
        seq2 = self.sent2seq(sentences2)
        score = self.model.evaluate([seq1, seq2], labels, batch_size=self.transform_batch)
        return score

    def load(self):
        # load all obj for predict
        print("loading tokenizer...")
        self.tokenizer = pickle.load(open(self.tokenizer_path, 'rb'))
        print("loading embedding_matrix...")
        self.embedding_matrix = np.load(self.embedding_matrix_path)
        print("loading two_textCNN_relation.h5...")
        self.model = load_model(self.bst_model_path)

    def load_model(self):
        # load h5 model for predict
        print("loading two_textCNN_relation.h5...")
        self.model = load_model(self.bst_model_path)


if __name__ == "__main__":
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



