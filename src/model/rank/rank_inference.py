import os, sys

join = os.path.join
dirname = os.path.dirname
sys.path.append(join(dirname(__file__), '..'))
sys.path.append(join(dirname(__file__), '../..'))
sys.path.append(join(dirname(__file__), '../../..'))
from config import PATH_ARGS

import torch
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import trange
from model.rank.rank_model import Net
from model.sentence_transformer import SentenceEmbedding

from time import time


class Inference:
    def __init__(self) -> None:
        self.model = Net(1536, 400, 2)
        model_path = join(PATH_ARGS.MODEL_DATA_DIR, 'sim_model_0421.pt')
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.sentence_embedding = SentenceEmbedding()

    def inference_sample(self,query, doc):
        query_embed, content_embed = self.sentence_embedding.get_emebeddings([query, doc])
        res = self.model(torch.Tensor(np.array([query_embed+content_embed])))
        with torch.no_grad():
            res = self.model(torch.Tensor([query_embed+content_embed]))
        res = res.detach().cpu().numpy()
        return res[0][1]

    def rank(self, query, docs):
        doc_ids = [i['doc_id'] for i in docs]
        titles = [i['title'] for i in docs]
        contents = [i['content'] for i in docs]
        nums = len(titles)
        new_text_set = [*titles, *contents, query]
        qnd_embeddings = self.sentence_embedding.get_emebeddings(new_text_set)
        query_embed, doc_embeds = qnd_embeddings[-1], qnd_embeddings[:-1]
        features = [query_embed+doc_embed for doc_embed in doc_embeds]
        with torch.no_grad():
            result = self.model(torch.Tensor(np.array(features)))
        result = result.detach().cpu().numpy()
        title_result, content_result = result[:nums], result[nums:]
        result = [[idx, title_score[1], content_score[1]] for idx, title_score, content_score in zip(doc_ids, title_result, content_result)]
        result = [[item[0], max(item[1], item[2])] if max(item[1], item[2])>=0.85 else [item[0], (item[1]+item[2])/2] for item in result]
        result = sorted(result, key=lambda x: x[1], reverse=True)
        result = {item[0]: item[1] for item in result}
        print(result)
        return result


if __name__ == "__main__":
    inferece = Inference()
    query = "对上海新冠防疫两措施的法律意见"
    content = {
        0: "对上海新冠防疫两措施的法律意见",
        1: "睡在上海浦东机场一个月",
        2: "当我决定离开上海",
        3: "童之伟：对上海新冠防疫两措施的法律意见。已披露的上海官方人员与相关居民的对话视频、音频显示，上海新冠病毒防疫采取的两项措施引起的事态非常严重，在市民中反应也很强烈，很可能造成某种法制灾难，特发表法律意见如下，以为各方处事的参考。",
        4: "上海人的忍耐已经到了极限。现在的上海人，每天晚上清点完冰箱忧心忡忡地睡下，每天早上抢完菜后忐忑不安地点开上海发布的数据，接着开启一天的核酸、抗原、团购、骂娘，以及求助。不知道为什么，每天都有刷新底线的事件。一位土生土长的上海居民，一个5岁孩子的爸爸，接受治疗后病情稳定的癌症患者，4月3日突感不适120去医院，在有前一天小区做的核酸报告情况下，被要求马上进行本院的核酸报告才可收治，在等待的过程中离开人世。他死前的最后一句话是“妈妈，你去问问医生，我的核酸报告出来了吗？”在他离去两个小时后，核酸报告出来了，阴性。",
        5: "【树 图 科 技 头 条】2022年5月10日 星期二, Conflux社区动态 1.【网络状态】Conflux网络算力≈1.5T，代币转移 TPS≈1130，昨日交易次数293K，新增账户214.20K，昨日新增合约13个。 2.【POS参数】POS总锁仓92M，节点总数148，年利率17.7%（理论计算），累计利息2.26M。 3.【海外项目】TransitFinance已加入Conflux eSpace生态系统，这是一个集成在TokenPocket_TP的聚合器。 4.【海外项目】TurboStarter和 Galaxy Blitz 将于 5 月 9 日至 5 月 12 日举办Giveaway活动，获奖者将获得 $MIT 代币和 Special NFT。 5.【生态项目】Conflux链上的原生稳定币项目（TriAngle）三角计划于5月13日，部署Conflux espace。 。 6.【社区动态】HydraSF矿池于已按5月1日零时的快照进行了两次矿池治理Token空投，，一次分发给质押HydraSF POS矿池的贡献者账户；第二次分发是对分发给参与锅第四次CFX生态基金处理方案投票的FC账户。‘ 7.【数字藏品】由烤仔潮物出品“烤仔–天选之子”数字藏品将于5月10日（今日）20：00上架Zverse（星图比特）. 8.【免责声明】本文仅代表个人观点，不构成投资建议，投资需谨慎。",
        7: "DeGate四月项目月报。欢迎来到DeGate 204月的项目月报！DeGate自己是以太坊上单点化的设备，并允许进行交易。DeGate 可以使用 DEX 所有的优势现货价单。买高卖”的策略。",
        8: "上海春天。在上海的第五年了，从未经历过这样的春天。前些年上学的时候，听说顾村公园樱花很不错，当时又懒又宅，不想约人又不想自己一个人去，磨来耗去竟从未去过。如今在宝山工作，顾村公园确实近了，但是由于新冠和政策，去一趟顾村公园看樱花竟成了更大的奢望。即使是愿意独自前往，也无法实现了。待到解封之时，还是多走走多逛逛吧。",
        9: "轮回之门。说不出来，肯定在哪里见过。“这是轮回之门！”柳无邪险些发出惊呼。轮回之门跟大轮回术，完全是两回事。修炼大轮回术，可以召唤出来轮回之路，斩断前世今生。而轮回之门，是镇守在轮回之路上的大门，谁能掌握轮回之门，等于掌握了轮回之路。以后面对神子，就算他施展大轮回法术，凭靠轮回之门，就能将其碾压。想要进入轮回，必须要经过这道门户。眼前看到的这座门户，只是一道虚影，并非真正的轮回之门。从记忆中得知，轮回之门早已消失亿万年，为何这里出现一道虚影，难道这里是一座轮回世界？短短半吸时间，柳无邪大脑出现无数信息。没有任何犹豫，一个纵射，消失的无影无踪，进入轮回之门。纳兰奇文等人迅速赶到，还是晚了一步，被柳无邪逃进去了。“我们怎么办？”纳兰秋禾一脸的愤怒之色，只有杀了柳无邪，她的道心才能圆满。这几日被柳无邪的意志，折磨的痛不欲生，每次闭上眼睛，柳无邪都会化身邪魔，钻入她的魂海。“追！”桃花门门主第一个追进去，丧子之痛，不共戴天，岂能让柳无邪活着离开千岛海域。越来越多的修士进入轮回之门。眨眼间的功夫，超过几万人进入其中。轮回之门依旧漂浮在空中，不过门户越来越不稳，出现晃动的迹象。千岛海域出现海市蜃楼的消息，越传越远，附近的岛屿修士，全部赶来。柳无邪进入轮回之门，仿佛进入另外一个世界。无尽的洪荒之力，充斥他的身体。“好古老的世界！”柳无邪进来之后，一个纵射，消失的无影无踪，以免被后面的人发现。大批修士进来，分散四周，寻找天地宝物。就在柳无邪进入不久，一名莽头人身的影子出现在海面上。“王，你为何坚持要找这个人？”从水底下又钻出来一尊恐怖的身体，似人非人，似妖非妖，长相有七分像人。““只有此人，才能帮助我们巫族复兴！”",
        10: "对上海新冠防疫两措施的法律意见。已披露的上海官方人员与相关居民的对话视频、音频显示，上海新冠病毒防疫采取的两项措施引起的事态非常严重，在市民中反应也很强烈，很可能造成某种法治灾难，特发表法律意见如下，以为各方处事的参考。"
    }
    res = inferece.rank(query, content)
    print(res)


