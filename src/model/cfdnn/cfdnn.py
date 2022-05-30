import os, sys
join = os.path.join
dirname = os.path.dirname
sys.path.append(join(dirname(__file__), '..'))

import torch
import numpy as np
import torch.nn as nn
from sklearn import metrics
from sentence_transformer import SentenceEmbedding
from transformers import get_linear_schedule_with_warmup


class CFDNN(nn.Module):
    def __init__(self, embed_dim, dropout_rate, learning_rate=1e-5, num_train_steps=20, device='cpu') -> None:
        super(CFDNN, self).__init__()
        self.learning_rate = learning_rate
        self.num_train_steps = num_train_steps
        self.gmf_layer = nn.Linear(embed_dim, int(embed_dim/2))
        self.dropout = nn.Dropout(dropout_rate)
        self.mlp_1 = nn.Linear(embed_dim*2, embed_dim)
        self.mlp_2 = nn.Linear(embed_dim, int(embed_dim/2))
        self.neu_mf = nn.Linear(embed_dim, 2)
        self.device = device

    def optimizer_scheduler(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        sch = get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=0,
            num_training_steps=self.num_train_steps,
        )
        # return optimizer and scheduler
        # optimizer is required and scheduler could be None or a scheduler object
        return opt, sch

    def loss(self, outputs, labels):
        if labels is None:
            return None
        return nn.CrossEntropyLoss()(outputs, labels)
    
    def monitor_metrics(self, outputs, labels):
        if labels is None:
            return {}
        outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1)
        labels = labels.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(labels, outputs)

        # return a acc dict, it can include several k-v pair and it'll be displayed in the description info of train progressbar
        return {"accuracy": torch.tensor(accuracy, device=self.device)}
    
    def forward(self, query, doc, labels=None): 
        cat_embed = torch.cat((query, doc), dim=1)
        gmf_input = query * doc
        # gmf_input = self.dropout(gmf_input)
        gmf_out = self.gmf_layer(gmf_input)
        gmf_out = torch.tensor(gmf_out, dtype=torch.float)
        cat_embed = torch.tensor(cat_embed, dtype=torch.float)

        # mlp layer
        # mlp_out = self.dropout(cat_embed)
        mlp_out = cat_embed
        mlp_out = self.mlp_1(mlp_out)
        # mlp_out = self.dropout(mlp_out)
        mlp_out = self.mlp_2(mlp_out)

        # concat mlp & gmf layer
        cat_gmf_mlp = torch.cat((gmf_out, mlp_out), dim=1)
        # cat_gmf_mlp = self.dropout(cat_gmf_mlp)
        output = self.neu_mf(cat_gmf_mlp)
        loss = self.loss(output, labels)
        acc = self.monitor_metrics(output, labels)
        return output, loss, acc


if __name__ == "__main__":
    se = SentenceEmbedding()
    cfdnn = CFDNN(768, 0.1)
    texts = ['今天天气好棒哦。', '今天天气真不错！']
    embeds = se.get_emebeddings(texts=texts)
    out = cfdnn(torch.LongTensor([embeds[0],embeds[1]]), torch.LongTensor([embeds[1],embeds[0]]))
    print(out)



