import os, sys

join = os.path.join
dirname = os.path.dirname
sys.path.append(join(dirname(__file__), '../..'))
from tqdm import trange
from utils.cos_simliarty import CosineSimilarity
from sentence_transformers import SentenceTransformer

class SentenceEmbedding:
    def __init__(self) -> None:
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        # self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.cs = CosineSimilarity()

    def get_emebeddings(self, texts, batch_size=64):
        if not isinstance(texts, list):
            texts = list(texts)
        embeddings = []
        # for st_i in trange(0, len(texts), batch_size, desc='get_embeddings'):
        for st_i in range(0, len(texts), batch_size):
            temp_texts = texts[st_i: st_i+64]
            temp_embeddings = self.model.encode(temp_texts)
            temp_embeddings = [list(i) for i in temp_embeddings]
            embeddings.extend(temp_embeddings)
        return embeddings

    def get_single_embedding(self, text):
        embeddings = self.model.encode([text])
        return embeddings[0]

# def cos_sim(vec1, vec2):
#     sim_score = cs.cos_similarity(vec1, vec2)
#     return sim_score

# if __name__ == "__main__":
#     # query = "钱理群“告别教育”"
#     # content = "任教五十年，钱理群在2012年教师节前夕宣布“告别教育”。从北大退休后，钱理群投身中学教育，试图“改变人心”，他以鲁迅自励，要在绝望中反抗，但基础教育十年试水，却令他收获“丰富的痛苦”。他说，—切不能为应试教育服务的教育根本无立足之地。"
#     # sim_score = cos_sim(*get_emebeddings([query, content]))
#     # print(sim_score)
#     sentence_embedding = SentenceEmbedding()
#     texts = ["钱理群“告别教育”", "从北大退休后，钱理群投身中学教育，试图“改变人心”"]
#     embeds = sentence_embedding.get_emebeddings(texts)
#     print(embeds)
#     print(len(embeds[0]))

