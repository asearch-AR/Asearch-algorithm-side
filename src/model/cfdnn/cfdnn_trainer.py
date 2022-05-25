import os
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from time import time
from tez import Tez, TezConfig
from copy import deepcopy as copy
from sklearn.utils import shuffle
from sklearn import model_selection
from transformers import AutoTokenizer
from tez.callbacks import EarlyStopping
from cfdnn import CFDNN
from sentence_transformer import SentenceEmbedding


os.environ["TOKENIZERS_PARALLELISM"] = "true"
join = os.path.join

class args:
    dataset_name = 'LCSTS_new'
    model_name = 'cfdnn'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 30
    batch_size = 32
    train_batch_size = 32
    valid_batch_size = 32
    max_len = 32
    random_states = 42
    accumulation_steps = 2
    dataloader_num_jobs = 1
    output_model_path = f"/mnt/disk/hdd1/arsearch/data/model/{dataset_name}/{model_name}_{str(int(time()))}.pt"
    train_data_path = f'/mnt/disk/hdd1/arsearch/data/data/{dataset_name}/train.tsv'
    feature_path = f'/mnt/disk/hdd1/arsearch/data/features/{dataset_name}'
    save_weights_only = True

class CustomDataset:
    def __init__(self, querys, docs, labels, name, reload=False):
        name_dict = {
            'train': 'train.pkl',
            'valid': 'valid.pkl',
            'test': 'test.pkl'
        }
        feature_path = join(args.feature_path, name_dict[name])
        self.querys = querys
        self.docs = docs
        self.labels = labels
        self.se = SentenceEmbedding()
        if os.path.exists(feature_path) and not reload:
            self.querys, self.docs = pkl.load(open(feature_path, 'rb'))
        else:
            self.querys = self.se.get_emebeddings(querys)
            self.docs = self.se.get_emebeddings(docs)
            feature = self.querys, self.docs
            pkl.dump(feature, open(feature_path, 'wb'))

    def __len__(self):
        return len(self.querys)

    def __getitem__(self, item):
        data_item = {
            "query": torch.tensor(self.querys[item]),
            "doc": torch.tensor(self.docs[item]),
            "labels": torch.tensor(self.labels[item], dtype=torch.long),
        }
        # print('data_item', data_item)
        return data_item

def train():
    # 1. Build a dataset
    ## load data
    dfx = pd.read_csv(args.train_data_path, error_bad_lines=False, sep='\t', header=0).fillna("none")
    # dfx = pd.read_excel(args.train_data_path, sheet_name='Sheet1', engine='openpyxl').fillna("none")
    dfx.columns = ['query', 'doc', 'label']
    dfx['label']= dfx['label'].astype('int')
    dfx = shuffle(dfx, random_state=args.random_states)
    ## split data
    df_train, df_valid = model_selection.train_test_split(
        dfx, test_size=0.3, random_state=args.random_states
    )
    df_test, df_valid = model_selection.train_test_split(
        df_valid, test_size=0.3, random_state=args.random_states
    )
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    df_test = df_valid.reset_index(drop=True)
    ## build a dataset
    train_dataset = CustomDataset(
        querys=df_train['query'].tolist(),
        docs=df_train['doc'].tolist(),
        labels=df_train['label'].tolist(),
        name='train'
    )
    valid_dataset = CustomDataset(
        querys=df_valid['query'].tolist(),
        docs=df_valid['doc'].tolist(),
        labels=df_valid['label'].tolist(),
        name='valid'
    )
    test_dataset = CustomDataset(
        querys=df_test['query'].tolist(),
        docs=df_test['doc'].tolist(),
        labels=df_test['label'].tolist(),
        name='test'
    )

    # 2. Build a model
    n_train_steps = int(len(train_dataset) / args.batch_size / args.accumulation_steps * args.epochs)
    model = CFDNN(768, 0.2)
    
    # 3. Build a trainer
    trainer_model = Tez(model)
    es = EarlyStopping(monitor="valid_loss", model_path=args.output_model_path, save_weights_only=args.save_weights_only)
    config = TezConfig(
        training_batch_size=args.train_batch_size,
        validation_batch_size=args.valid_batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        epochs=args.epochs,
        step_scheduler_after="epoch",
        device=args.device,
        num_jobs=args.dataloader_num_jobs
    )

    # 4. Train
    trainer_model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        callbacks=[es],
        config=config,
    )

def load_model(model_path=args.output_model_path):
    model = CFDNN()

    # 是否仅加载模型参数
    if args.save_weights_only:
        model.load_state_dict(torch.load(model_path, map_location=args.device))
    else:
        model.load_state_dict(torch.load(model_path, map_location=args.device)["state_dict"])
    return model

def test(model, texts, batch_size=64):
    res = []
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for batch_idx in range(0, len(texts), batch_size):
        part_texts = texts[batch_idx: batch_idx+batch_size]
        ids, masks, token_type_ids = [], [], []
        for text in part_texts:
            inputs = tokenizer.encode_plus(
                    text,
                    None,
                    add_special_tokens=True,
                    max_length=args.max_len,
                    padding="max_length",
                    truncation=True
                )
            ids.append(inputs["input_ids"])
            masks.append(inputs["attention_mask"])
            token_type_ids.append(inputs["token_type_ids"])
        with torch.no_grad():
            output = model(ids=torch.LongTensor(ids), mask=torch.LongTensor(masks), token_type_ids=torch.LongTensor(token_type_ids))[0]
            output = np.argmax(output.detach().numpy(), axis=1)
            res.extend(output)
    return res


if __name__ == "__main__":
    train()