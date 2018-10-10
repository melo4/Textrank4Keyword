# -*- encoding:utf-8 -*-

import os
import math
import networkx as nx
import numpy as np
import sys

sentence_delimiters = ['?','!',';','…','……','。','！','？','；','\n']
allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz','v', 't', 'vd', 'vn', 'eng']

__DEBUG = None

def debug(*args):
    global __DEBUG
    if __DEBUG is None:
        try:
            if os.environ['DEBUG'] == '1':
                __DEBUG = True
            else:
                __DEBUG = False
        except:
            __DEBUG = False
    if __DEBUG:
        print(' '.join([str(arg) for arg in args]))

class AttrDict(dict):
    """Dict that can get attribute by dot"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def combine(word_list, window=2):
    """
    :param word_list: 由单词组成的列表
    :param window: int  窗口大小
    :return:
    """
    if window < 2:
        window = 2
    for x in range(1, window):
        if x >= len(word_list):
            break
        word_list2 = word_list[x:]
        res = zip(word_list,word_list2)
        for r in res:
            yield r

def get_similarity(word_list1, word_list2):
    """
    获取两个句子之间的相似度
    :param word_list1:
    :param word_list2: 都是单词组成的列表，分表代表一个句子。
    :return:
    """
    words = list(set(word_list1 + word_list2))
    vector1 = [float(word_list1.count(word)) for word in words]
    vector2 = [float(word_list2.count(word)) for word in words]

    vector3 = [vector1[x] * vector2[x] for x in range(len(vector1))]
    vector4 = [1 for num in vector3 if num > 0.]
    co_occur_num = sum(vector4)

    if abs(co_occur_num) <= 1e-12:
        return 0.

    denominator = math.log(float(len(word_list1))) + math.log(float(len(word_list2)))  # 分母
    if abs(denominator) < 1e-12:
        return 0.

    return co_occur_num / denominator

def sort_words(vertex_source, edge_source, window=2, pagerank_config={'alpha':0.85,}):
    """
    将单词按关键程度从大到小排序
    :param vertex_source: 二维列表，子列表代表句子，子列表的元素是单词，单词用来构造pagerank中的节点
    :param edge_source: 二维列表，子列表代表句子，子列表的元素是单词，根据单词位置关系构造pagerank中的边
    :param window: 一个句子中相邻的window个单词，两两之间被认为有边
    :param pagerank_config: pagerank中的设置
    :return:
    """
    sorted_words = []
    word_index = {}
    index_word = {}
    _vertex_source = vertex_source
    _edge_source = edge_source
    words_number = 0

    for word_list in _vertex_source:
        for word in word_list:
            if not word in word_index:
                word_index[word] = words_number
                index_word[words_number] = word
                words_number += 1

    graph = np.zeros((words_number,words_number))

    for word_list in _edge_source:
        for w1, w2 in combine(word_list, window):
            if w1 in word_index and w2 in word_index:
                index1 = word_index[w1]
                index2 = word_index[w2]
                graph[index1][index2] = 1.0
                graph[index2][index1] = 1.0

    debug('graph:\n', graph)

    nx_graph = nx.from_numpy_matrix(graph)
    scores = nx.pagerank(nx_graph, **pagerank_config)   # this is a dict
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for index, score in sorted_scores:
        item = AttrDict(word=index_word[index], weight=score)
        sorted_words.append(item)
    return sorted_words

def sort_sentences(sentences, words, sim_func = get_similarity, pagerank_config={'alpha':0.85,}):
    """
    将句子按照关键程度从大到小排序
    :param sentences: 列表，元素是句子
    :param words: 二维列表，子列表和sentence中的句子对应，子列表由单词组成
    :param sim_func: 相似性
    :param pagerank_config: pagerank的设置
    :return:
    """
    sorted_sentences = []
    _source = words
    sentences_num = len(_source)
    graph = np.zeros((sentences_num, sentences_num))

    for x in range(sentences_num):
        for y in range(x, sentences_num):
            similarity = sim_func(_source[x], _source[y])
            graph[x, y] = similarity
            graph[y, x] = similarity

    nx_graph = nx.from_numpy_matrix(graph)
    scores = nx.pagerank(nx_graph, **pagerank_config)  # this is a dict
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    for index, score in sorted_scores:
        item = AttrDict(index=index, sentence=sentences[index], weight=score)
        sorted_sentences.append(item)

    return sorted_sentences

if __name__ == '__main__':
    pass
