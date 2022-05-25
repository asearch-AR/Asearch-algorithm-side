import os, sys

join = os.path.join
dirname = os.path.dirname
sys.path.append(join(dirname(__file__), '..'))
sys.path.append(join(dirname(__file__), '../..'))
from config import PATH_ARGS

import torch
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import trange
from rank_model import Net
from copy import deepcopy as copy
from sklearn.utils import shuffle
from sklearn.metrics import log_loss
from sentence_transformer import SentenceEmbedding
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def process_data(data_path):
    sentence_embedding = SentenceEmbedding()
    data = pd.read_csv(data_path, sep='\t', header=0)
    data = shuffle(data, random_state=0)
    summary_embeds, content_embeds = [], []
    batch_size = 256
    for batch_idx in trange(0, len(data), batch_size, desc='get_emebeddings'):
        batch_summary = data.summary.tolist()[batch_idx: batch_idx+batch_size]
        batch_content = data.content.tolist()[batch_idx: batch_idx+batch_size]
        batch_summary_embeds = sentence_embedding.get_emebeddings(batch_summary)
        batch_content_embeds = sentence_embedding.get_emebeddings(batch_content)
        summary_embeds.extend(batch_summary_embeds)
        content_embeds.extend(batch_content_embeds)
    embeddings = [list(s)+list(c) for s, c in zip(summary_embeds, content_embeds)]
    features = embeddings, data.label.tolist()
    output_path = data_path.split('.')
    output_path[-1] = 'pkl'
    output_path = '.'.join(output_path)
    pkl.dump(features, open(output_path, 'wb'))

def train_model(X, Y, model, model_path):
    X = torch.Tensor(X)
    Y = torch.LongTensor(Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=0)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_func = torch.nn.CrossEntropyLoss()
    steps = 400
    batch_size = 128
    loss = 0
    for step in trange(steps, desc='train'):
        model.train()
        for batch_index in range(0, len(x_train), batch_size):
            x = x_train[batch_index:batch_index+batch_size]
            y = y_train[batch_index:batch_index+batch_size]
            out = model(x)
            loss = loss_func(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if step % 50 ==0:
            model.eval()
            y_pred = model(x_test).detach().numpy()
            eval_loss = log_loss(y_test, y_pred, labels=range(2))
            y_pred = np.argmax(y_pred, axis=1)
            f1_report = classification_report(y_test, y_pred, target_names=['0', '1'])
            print('eval_loss', eval_loss)
            print(f1_report)
        if step % 200 == 0:
            torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    input_path = join(PATH_ARGS.MAIN_DATA_DIR, 'LCSTS_new', 'train.tsv')
    process_data(input_path)
    feature_path = join(PATH_ARGS.MAIN_DATA_DIR, 'LCSTS_new', 'train.pkl')
    feature = pkl.load(open(feature_path, 'rb'))
    X, Y = feature
    input_dim = len(X[0])
    print(input_dim)
    model = Net(input_dim, 400, 2)
    model_path = join(PATH_ARGS.MODEL_DATA_DIR, 'sim_model_0421.pt')
    train_model(X, Y, model, model_path)
