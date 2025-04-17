
#1. 使用豆瓣电影评论数据完成文本分类处理：文本预处理，加载、构建词典。（评论得分1～2 表示negative取值：1，评论得分4～5代表positive取值：0）
# https://www.kaggle.com/datasets/utmhikari/doubanmovieshortcomments
#2. 加载处理后文本构建词典、定义模型、训练、评估、测试。
#3. 尝试不同分词工具进行文本分词，观察模型训练结果。

import pickle
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence  # 长度不同张量填充为相同长度
import jieba

#定义一个将分词结果转换为 {word: index}的函数
def build_vocab(doc):
    vocab = set()
    for line in doc:
        vocab.update(line[0])
    vocab =  ['PAD','UNK'] + list(vocab)
    w_index = {word: idx for idx, word in enumerate(vocab)}
    return w_index



#1、加载评论分词完成的数据
with open('ds_comments.pkl','rb') as f:
    comments_data = pickle.load(f)
    print(comments_data[0])
        
    