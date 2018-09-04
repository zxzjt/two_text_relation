#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#-------------------------------------------------------------------------------
'''
@Author   :   zxzjt
@Contact  :   zhujingtao@loonxi.com
@License  :   (C) Copyright 2013-2018, loonxi
@Software :   PyCharm Community Edition
@File     :   text_preprocessing.py
@Created  :   2018/7/6 20:19
@Version  :   0.0.1
@Desc     :   text preprocess, modified for two_textCNN_relation model
去除非英文字符，特殊字符，恶意字符，表情符号
提取某些特殊词语（提前提取特殊缩写分隔符，numbel，whatsapp，以免后面被去除）
去除超链接，邮箱
分句，分词
去除标点符号
手机号，数量等字符处理（可去除）
转换为小写
英文拼写缩写还原（可选）（需人为建立拼写缩写还原库字典）
拼写检查（）
添加词性特征nltk.pos_tag（可选）
词干提取nltk.stem，词形还原(先小写)
去掉长度过小的词（长度小于2，可选）
停用词
自定义停词

注意：保留白名单词库（建立白名单词库，该词库避免拼写检查，避免长度过滤，保留停用词，
包括人为添加的拼写缩写还原库字典，命名实体库，保留停用词库，保留短词库-->无需拼写检查库）

工具：
正则模块re，过滤异常字符
分词、分句（nltk中nltk.word_tokenize和sent_tokenizer.tokenize）
（nltk.regexp_tokenize，WordPunct tokenizer，TreebankWordTokenizer，和WhitespaceTokenizer）
单词拼写错误pyenchant，Pattern.en
词性特征nltk.pos_tag，nltk.corpus.brown，TextBlob，Polyglot，MontyLingua
词干提取map和列表推导式（Porter stemmer, Lancaster Stemmer，WordNetLemmatizer和 Snowball Stemmer）
词形还原map(stemmer.stem, raw_tokens)，[stemmer.stem(token) for token in raw_tokens]
过滤停词函数filter()，nltk.corpus.stopwords
建立词典（collections，from tensorflow.contrib import learn.preprocessing.VocabularyProcessor，from gensim.corpora import
Dictionary，from sklearn.feature_extraction.text import CountVectorizer，keras.preprocessing.text fit_on_text
Tokenizer,keras.preprocessing.text.texts_to_sequences）
文本填充（keras.preprocessing.sequence.pad_sequences，from tflearn.data_utils import pad_sequences）
'''
#-------------------------------------------------------------------------------

"""Code Base
千岩万转路不定，迷花倚石忽已暝
"""

import re

try:
    import enchant
except:
    has_enchant = False
else:
    has_enchant = True

from pattern3.text import en
from nltk.tokenize import WhitespaceTokenizer,TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from .global_log import logger_debug

class TextCleaner(object):
    """文本清理
    去除非英文字符，提取特殊缩写，去除邮箱、url，去除数字
    """
    def __init__(self):
        # 需单独分开的缩写
        self.apostrophe = ["\'s", "\'ve", "n\'t", "\'re","\'d","\'ll"]
        self.norm_apostrophe = ["\'s", " have ", " not ", " are ", " would ", " will "]

    def _clean_spec(self, string):
        # 去除非英文字符
        new_string = re.sub(r"[^A-Za-z0-9()<>,:;=+_*./!?@$%&|{}\[\]\-\'\"\`]", " ", string)
        return new_string

    def __extract_others(self, string):
        # 提取no.为number
        new_string = re.sub('no\.? ?[0-9]{1,18}[0-9\-]{0,18}', 'number', string)
        # 提取whatapp
        new_string = re.sub('what *\'?s? *app', 'whatsapp', new_string)
        return new_string

    def _extract_apostrophe(self, string):
        # 提取特殊缩写
        new_string = string
        for i,apo in enumerate(self.apostrophe):
            new_string = re.sub(apo, self.norm_apostrophe[i], new_string)
        # 提取其他
        new_string = self.__extract_others(new_string)
        return new_string

    def _clean_email(self, string):
        # 去除邮箱
        new_string = re.sub(r'[_\-\w\.]{0,19}@[-_\w\.]{1,13}\.[a-z]{1,3}', ' ', string)
        return new_string

    def _clean_url(self, string):
        # 去除url
        new_string = re.sub('https?://[.?=%&/\w\-]{2,100}', ' ', string)
        return new_string
    """
    def _clean_number(self, string):
        # 去除数字和字符混合字符串
        #new_string = re.sub('[a-zA-Z]{1,18}[0-9\-]{1,18}', ' ', string)
        # 去除数字
        new_string = re.sub('[0-9]{1,18}[0-9\-]{0,18}', ' ', string)
        return new_string
    """
    def clean_string(self, string):
        # 集成处理
        new_string = self._extract_apostrophe(self._clean_spec(string))
        new_string = self._clean_url(self._clean_email(new_string))
        #new_string = self._clean_number(new_string)
        return new_string

class TextTokenizer(object):
    """分词
    去除标点符号，保留@,$,&,'，TreebankWordTokenizer分词
    """
    def __init__(self):
        self.tokenizer = TreebankWordTokenizer()
        self.puncts = [] # 需保留的字符@,$,&

    def _clean_punct(self, string):
        # 去除标点符号
        new_string = re.sub(r'[,.;+=<>()/:_?!$@&%*|{}\-\[\]\"\']', ' ', string) # 保留@,$,&
        return new_string

    def _extract_punct(self, string):
        # 保留@,$,&
        """
        new_string = string
        for punct in self.puncts:
            if punct == '$':
                new_string = re.sub('\$', ' ' + punct + ' ', new_string)
            else:
                new_string = re.sub(punct, ' '+punct+' ', new_string)
        return new_string
        """
        return string

    def _tokenize(self, string):
        # TreebankWordTokenizer分词
        word_list = self.tokenizer.tokenize(string)
        return word_list

    def tokenize(self, string):
        # 集成处理
        new_string = self._extract_punct(self._clean_punct(string))
        word_list = self._tokenize(new_string)
        return word_list

class WordsProcessor(object):
    """分词后单词处理
    缩写拼写检查，词形还原，过滤停词
    """
    def __init__(self, redefined_stop_words, reserved_words, abbr_dict, spelling_suggestion = False):
        self.lemmatizer = WordNetLemmatizer()
        self.d = enchant.Dict("en") if has_enchant else None
        # 常用缩写字典库
        self.abbr_dict = abbr_dict
        # spelling 开启
        self.spelling_suggestion = spelling_suggestion
        #  生成stop words
        stop_words = stopwords.words('english')
        self.reserved_words = ['no', 'not', "n't"]+reserved_words+list(self.abbr_dict.values())
        self.stop_words = [word for word in stop_words if word not in self.reserved_words] + redefined_stop_words

    def _filter_number(self, words):
        # 过滤数字开头字符串
        #words_filter = list(filter(lambda w: not w.isdigit(), words))
        words_filter = [re.sub("^\d.*", '', word) for word in words]
        words_filter = [word.strip() for word in words_filter if word.strip() != '']
        return words_filter

    def __spelling_reduction(self, words):
        # 人为添加拼写还原
        words_reduced = [self.abbr_dict.get(word,word) for word in words]
        return words_reduced

    def _abbreviation_reduction(self, words):
        # 先小写，再缩写还原
        words_lower = [word.lower() for word in words]
        words_reduced = self.__spelling_reduction(words_lower)
        return words_reduced

    def _spelling_suggest(self, words, mode = 'pattern3'):
        """拼写检查
        spelling suggest
        :param words:
        :param mode:
        :return:
        """
        if self.spelling_suggestion:
            words_checked = []
            for word in words:
                if word not in self.reserved_words:
                    if mode == 'enchant':
                        # use enchant
                        words_checked.append(self.d.suggest(word)[0])
                    else:
                        # use pattern3.en.suggest
                        words_checked.append(en.suggest(word)[0][0])
                else:
                    words_checked.append(word)
            return words_checked
        else:
            return words

    def _lemmatizer(self, words):
        # 先小写，词形还原
        #if w not in self.reserved_words:
        words_lemmatized = list(map(lambda w: self.lemmatizer.lemmatize(w, 'n') if w not in self.reserved_words else w, words))
        #words_lemmatized = list(map(lambda w: self.lemmatizer.lemmatize(w, 'v') if w not in self.reserved_words else w, words_lemmatized))
        return words_lemmatized

    def _clean_short(self, words, reserved_words):
        # 去除长度小于2的word
        words_long = list(filter(lambda word:len(word) > 1 or word in reserved_words,words))
        return words_long

    def _filter_stopwords(self, words):
        # 过滤停词
        words_filter = list(filter(lambda word: word not in self.stop_words, words))
        return words_filter

    def words_process(self, words):
        # 集成处理
        words_processed = self._spelling_suggest(self._abbreviation_reduction(self._filter_number(words)))
        words_processed = self._clean_short(self._lemmatizer(words_processed), reserved_words=self.reserved_words)
        words_processed = self._filter_stopwords(words_processed)
        return words_processed

# 手动添加停词
enquiry_redefined_stop_words = []
enquiry_named_entity = []
enquiry_spelling_abbr_dict = {'pp': 'pp', 'pm': 'pm', 'rfq': 'rfq', 'whatsup': 'whatsapp', 'Watssup': 'whatsapp',
                            'alibsba': 'alibaba', 'bro': 'brother', 'buing': 'buying', 'buzness': 'business',
                            'cll': 'call', 'cntct': 'contact', 'app': 'whatsapp', 'plz': 'please', 'tshirts': 'tshirt',
                            'pls': 'please', 'watsup': 'whatsapp', 'msg': 'message', 'cont': 'contact',
                            'con': 'contact', 'whatsap': 'whatsapp', 'led': 'led', 'grp': 'group', 'wats': 'whatsapp',
                            'mackook': 'macbook', 'mackooks': 'macbook', 'mackbook': 'macbook', 'mackbooks': 'macbook',
                            'macbooks': 'macbook','sundries':'sundry','eskayinternational':'eskay','decor':'decoration',
                            'iphones': 'iphone', 'wtsup': 'whatsapp', 'watsapp': 'whatsapp', 'chck': 'check',
                            'watsap': 'whatsapp','amp ':'amplifier','grup':'group','evening':'evening','abs':'abs',
                            'whtsapp': 'whatsapp', 'wtsapp': 'whatsapp', 'whtsap': 'whatsapp', 'whts': 'whatsapp',
                            'whtsup': 'whatsapp','watspp': 'whatsapp', 'wattssup': 'whatsapp'}

class TextProcessor(object):
    """集成文本预处理
    文本清理，分词
    """
    def __init__(self, redefined_stop_words = enquiry_redefined_stop_words, named_entity = enquiry_named_entity, abbr_dict = enquiry_spelling_abbr_dict, spelling_suggestion = False):
        self.text_cleaner = TextCleaner()
        self.text_tokenizer = TextTokenizer()
        self.words_processor = WordsProcessor(redefined_stop_words, self.text_tokenizer.puncts+named_entity, abbr_dict, spelling_suggestion)

    def text_process(self, string, res='words'):
        # 集成文本预处理
        try:
            new_string = self.text_cleaner.clean_string(string)
        except:
            logger_debug.exception("text_cleaner.clean_string Exception")
            return [] if res=='words' else ''
        else:
            try:
                words = self.text_tokenizer.tokenize(new_string)
            except:
                logger_debug.exception("text_tokenizer.tokenize Exception")
                return [] if res=='words' else ''
            else:
                try:
                    words = self.words_processor.words_process(words)
                except:
                    logger_debug.exception("words_processor.words_process Exception")
                    return [] if res=='words' else ''
                else:
                    new_string = ' '.join(words)
                    return words if res == 'words' else new_string


if __name__=="__main__":
    print(TextProcessor().text_process("Baby girls clothes nb and 3  45-89 audi a4 a445abd558erf"))
    """
    df_sampled = read_file('./micro_enquiry_post_comment_cleaned.csv')
    df_sampled = [string.split("\t") for string in df_sampled]
    micro_enquiry_docs = [x[3].replace('\n','') for x in df_sampled]
    """

