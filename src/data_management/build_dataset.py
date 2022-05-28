import os, sys

join = os.path.join
dirname = os.path.dirname
sys.path.append(join(dirname(__file__), '..'))
sys.path.append(join(dirname(__file__), '../..'))

import json
import jieba 
import pandas as pd
from tqdm import trange
from config import PATH_ARGS
from sklearn.utils import shuffle
from model.simbert import get_emebeddings 
from utils.cos_simliarty import CosineSimilarity
from sklearn.feature_extraction.text import TfidfVectorizer


def extra_keywords(corpus):
    tfidf_vec = TfidfVectorizer()
    tfidf_matrix = tfidf_vec.fit_transform(corpus)
    feature_names = tfidf_vec.get_feature_names()
    corpus_keywords = []
    for i in tfidf_matrix.toarray():
        dict_i = {j: k for j, k in zip(feature_names, i)}
        item_top_keywords = sorted(dict_i.items(), key=lambda x: x[1], reverse=True)[:3]
        item_top_keywords = [i[0] for i in item_top_keywords]
        corpus_keywords.append(''.join(item_top_keywords))
    return corpus_keywords

def process_summary_data():
    cs = CosineSimilarity()
    summary_train_data_path = join(PATH_ARGS.MAIN_DATA_DIR, 'LCSTS_new', 'train.json')
    summary_train_data = [json.loads(i.strip()) for i in open(summary_train_data_path, 'r').readlines()[:40000]]
    summary_train_data = [[i['summary'], i['content']] for i in summary_train_data]
    pos_samples = list(zip(*summary_train_data[:20000]))
    pos_summary = pos_samples[0]
    neg_samples_base = list(zip(*summary_train_data[20000:]))
    neg_summary = neg_samples_base[0]
    pos_summary_embedding, neg_summary_embedding = [], []

    batch_size = 256
    for batch_idx in trange(0, len(pos_summary), batch_size, desc='pos_embedding'):
        batch_embeds = get_emebeddings(pos_summary[batch_idx: batch_idx+batch_size])
        pos_summary_embedding.extend(batch_embeds)
    
    for batch_idx in trange(0, len(neg_summary), batch_size, desc='neg_embedding'):
        batch_embeds = get_emebeddings(neg_summary[batch_idx: batch_idx+batch_size])
        neg_summary_embedding.extend(batch_embeds)
    
    cos_sims = cs.cos_similarity_matrix(neg_summary_embedding, pos_summary_embedding)
    neg_new_summary = []
    for line in cos_sims:
        line = list(enumerate(line))
        line = sorted(line, key=lambda x: x[1], reverse=True)
        best_neg_sample = ''
        for ix, score in line:
            if score < 0.5:
                best_neg_sample = pos_summary[ix]
                break
        if best_neg_sample == '':
            best_neg_sample = pos_summary[line[-1][0]]
        neg_new_summary.append(best_neg_sample)
    neg_samples = [neg_new_summary, list(neg_samples_base[1]), [0]*len(neg_new_summary)]
    pos_samples = [list(pos_samples[0]), list(pos_samples[1]), [1]*len(pos_samples[0])]
    samples = [p+n for p, n in zip(pos_samples, neg_samples)]
    samples = list(zip(*samples))
    
    # keywords match
    corpus = [' '.join(jieba.lcut(i[1])) for i in samples]
    corpus_keywords = extra_keywords(corpus)
    query = [' '.join(jieba.lcut(i[0])) for i in samples]
    query_keywords = extra_keywords(query)

    aug_samples = []
    for item, c, q in zip(samples, corpus_keywords, query_keywords):
        aug_samples.append(item)
        aug_samples[-1][0] = c
        aug_samples.append(item)
        aug_samples[-1][0] = q
    
    samples += aug_samples

    samples = pd.DataFrame(samples, columns=['summary', 'content', 'label'])
    sample_length = len(samples)/2

    lcqmc_data_path = join(PATH_ARGS.MAIN_DATA_DIR, 'lcqmc', 'train.tsv')
    lcqmc_data = pd.read_csv(lcqmc_data_path, sep='\t', names=['summary', 'content', 'label'])
    pos_lcqmc_data = lcqmc_data[lcqmc_data['label']==1].iloc[:sample_length]
    neg_lcqmc_data = lcqmc_data[lcqmc_data['label']==0].iloc[:sample_length]

    samples = pd.concat([samples, pos_lcqmc_data, neg_lcqmc_data])
    samples = shuffle(samples, random_state=42)

    output_path = join(PATH_ARGS.MAIN_DATA_DIR, 'train.tsv')
    samples.to_csv(output_path, index=False, sep='\t')


if __name__ == '__main__':
    process_summary_data()
