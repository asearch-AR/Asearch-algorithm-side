import os, sys

join = os.path.join
dirname = os.path.dirname
sys.path.append(join(dirname(__file__), '..'))
sys.path.append(join(dirname(__file__), '../..'))

from tqdm import trange
from model.sentence_transformer import get_emebeddings


def extract_feature(query_content_pairs, batch_size=128):
    query_content_pairs = sum(query_content_pairs, [])
    query_content_embeds = []
    for batch_idx in trange(0, len(query_content_pairs), batch_size, desc='extract_feature'):
        temp_text = query_content_pairs[batch_idx: batch_idx+batch_size]
        temp_embed = get_emebeddings(temp_text)
        query_content_embeds.extend(temp_embed)
    features = []
    for f_idx in range(0, len(query_content_embeds), 2):
        features.append(
            query_content_embeds[f_idx]+query_content_embeds[f_idx+1]
        )
    return features

