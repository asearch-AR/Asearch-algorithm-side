#！-*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import *
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
from bert4keras.models import build_transformer_model
from tensorflow.python.keras.backend import set_session

os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

sess = tf.Session()
graph = tf.get_default_graph()

maxlen = 32

# 建立分词器
tokenizer = Tokenizer(Simbert_Args.dict_path, do_lower_case=True)  # 建立分词器

set_session(sess)

# 建立加载模型
bert = build_transformer_model(
    Simbert_Args.config_path,
    Simbert_Args.checkpoint_path,
    with_pool='linear',
    application='unilm',
    return_keras_model=False,
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])

def get_emebeddings(texts):
    """
        计算句向量
    """
    text_length = 0
    # print(texts)
    for i in texts:
        text_length = max(text_length, len(i))
    token_ids_list = []
    for text in texts:
        token_ids = tokenizer.encode(text, max_length=maxlen)[0]
        token_ids_list.append(token_ids)
    token_ids_list = sequence_padding(token_ids_list)
    # print(token_ids_list)
    global sess
    global graph
    global encoder
    with graph.as_default():
        set_session(sess)
        vectors = encoder.predict([token_ids_list, np.zeros_like(token_ids_list)])
        return vectors.tolist()
        # return [list(item) for item in vectors]

def get_single_emebedding(text):
    """
        计算句向量
    """
    token_ids, segment_ids = tokenizer.encode(text)
    global sess
    global graph
    global encoder
    with graph.as_default():
        set_session(sess)
        vec = encoder.predict([[token_ids], [segment_ids]])
        return vec[0]

# print(simbert_vec(['你知道如何祛斑', '你知道如何祛斑吗哈哈哈']))