import grpc
from proto import asearch_pb2, asearch_pb2_grpc


if __name__ == '__main__':
    with grpc.insecure_channel('localhost:50000') as channel:
        stub = asearch_pb2_grpc.ASearchSimilarityStub(channel=channel)
        query = "钱理群“告别教育”"
        docs = {
            '1': "任教五十年，钱理群在2012年教师节前夕宣布“告别教育”。从北大退休后，钱理群投身中学教育，试图“改变人心”，他以鲁迅自励，要在绝望中反抗，但基础教育十年试水，却令他收获“丰富的痛苦”。他说，—切不能为应试教育服务的教育根本无立足之地。",
            '2': "任教五十年，钱理群在2012年教师节前夕宣布“告别教育”。",
            '3': "从北大退休后，钱理群投身中学教育，试图“改变人心”，",
            '4': "他说，—切不能为应试教育服务的教育根本无立足之地。",
            '5': "任教五十年，钱理群在2012年教师节前夕宣布“告别教育”。从北大退休后，钱理群投身中学教育，试图“改变人心”，他以鲁迅自励，要在绝望中反抗，但基础教育十年试水，却令他收获“丰富的痛苦”。他说，—切不能为应试教育服务的教育根本无立足之地。",
            '6': "任教五十年，钱理群在2012年教师节前夕宣布“告别教育”。",
            '7': "从北大退休后，钱理群投身中学教育，试图“改变人心”，",
            '8': "他说，—切不能为应试教育服务的教育根本无立足之地。"
        }
        rsp: asearch_pb2.RankResponse = stub.ReRank(asearch_pb2.RankRequest(query=query, docs=docs))
        print(rsp.result)

        query = "钱理群“告别教育”"
        doc = "任教五十年，钱理群在2012年教师节前夕宣布“告别教育”。从北大退休后，钱理群投身中学教育，试图“改变人心”，他以鲁迅自励，要在绝望中反抗，但基础教育十年试水，却令他收获“丰富的痛苦”。他说，—切不能为应试教育服务的教育根本无立足之地。"
        rsp: asearch_pb2.SimResponse = stub.TextSimilarty(asearch_pb2.SimRequest(query=query, doc=doc))
        print(rsp.sim_score)

        texts = ["钱理群“告别教育”", "从北大退休后，钱理群投身中学教育，试图“改变人心”"]
        rsp: asearch_pb2.EmbedResponse = stub.GetEmbedding(asearch_pb2.EmbedRequest(texts=texts))
        print(len(rsp.embeddings))

        