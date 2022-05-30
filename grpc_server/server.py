import os, sys

from numpy import e
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import grpc
from concurrent import futures
from grpc_server.proto import asearch_pb2, asearch_pb2_grpc
from src.model.rank.rank_inference import Inference
from src.model.sentence_transformer import SentenceEmbedding

class ASearchSimilarity(asearch_pb2_grpc.ASearchSimilarityServicer):
    def __init__(self) -> None:
        super().__init__()
        self.inference = Inference()
        self.sentence_embedding = SentenceEmbedding()

    def ReRank(self, request, context):
        query = request.query
        docs = [{'doc_id': item.doc_id, 'title': item.title, 'content': item.content} for item in request.docs]
        result = self.inference.rank(query, docs)
        return asearch_pb2.RankResponse(result=result)
    
    def TextSimilarty(self, request, context):
        sim_score = self.inference.inference_sample(request.query, request.doc)
        return asearch_pb2.SimResponse(sim_score=sim_score)
    
    def GetEmbedding(self, request, context):
        embeddings = self.sentence_embedding.get_emebeddings(request.texts)
        embeddings = [asearch_pb2.Embedding(embedding=i) for i in embeddings]
        return asearch_pb2.EmbedResponse(embeddings=embeddings)


if __name__ == '__main__':
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    asearch_pb2_grpc.add_ASearchSimilarityServicer_to_server(ASearchSimilarity(), server)
    server.add_insecure_port('0.0.0.0:50000')
    server.start()
    print('running')
    server.wait_for_termination()