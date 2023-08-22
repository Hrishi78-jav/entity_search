import pandas as pd
from sentence_transformers import InputExample
from tqdm import tqdm
import torch
import pickle
from torch.utils.data import DataLoader
# from datasets import load_dataset
import time
import csv
from itertools import chain, cycle


def convert_2_InputExample_split(df):
    data = []
    n_examples = len(df)

    for i in tqdm(range(n_examples)):
        example = df.iloc[i]
        data.append(InputExample(texts=[example['Query'], example['Positive']],
                                 label=1))

    # random.shuffle(data)  #shuffle then split

    train = data[:int(0.98 * n_examples)]
    val = data[int(0.98 * n_examples):int(0.99 * n_examples)]
    test = data[int(0.99 * n_examples):]
    # train = data[:int(0.8*n_examples)]
    # val = data[int(0.8*n_examples):int(0.9*n_examples)]
    # test = data[int(0.9*n_examples):]

    print(f"Train Data Length = {len(train)}")
    print(f"Validation Data Length = {len(val)}")
    print(f"Test Data Length = {len(test)}")
    return train, val, test


def build_data():
    df1 = pd.read_csv('../data/Synthetic_8M_clean.csv', chunksize=5000000)  # 5M clean
    df2 = pd.read_csv('../data/Set1.csv', chunksize=1000000)  # rest 5M misspells
    df3 = pd.read_csv('data/Set2.csv', chunksize=1000000)
    df4 = pd.read_csv('data/Set3.csv', chunksize=1000000)
    df5 = pd.read_csv('data/Set4.csv', chunksize=1000000)
    df6 = pd.read_csv('data/Set5.csv', chunksize=1000000)
    dff2, dff3, dff4, dff5, dff6 = next(df2), next(df3), next(df4), next(df5), next(df6)
    df = pd.concat([df1, dff2, dff3, dff4, dff5, dff6])
    df.to_csv('data/train_data_10M.csv', index=False)


# class Iterable_dataset2(torch.utils.data.IterableDataset):
#     def __init__(self, path="data/sample_val.csv", length=1000):
#         super().__init__()
#         self.data = load_dataset("csv", data_files=path, split="train", streaming=True)
#         self.length = length
#
#     def __len__(self):
#         return self.length
#
#     def __iter__(self):
#         return iter(self.data)

class Map_dataset(torch.utils.data.Dataset):
    def __init__(self, path="data/sample_val.csv"):
        super().__init__()
        self.df = pd.read_csv(path)

    def __len__(self):
        return len(self.df)

    def process_data(self, idx):
        query, positive = self.df['Query'].iloc[idx], self.df['Positive'].iloc[idx]
        return InputExample(texts=[query, positive], label=1)

    def __getitem__(self, idx):
        return self.process_data(idx)


class Iterable_dataset(torch.utils.data.IterableDataset):
    def __init__(self, path="data/sample_val.csv", length=1000):
        super().__init__()
        self.path = path
        self.length = length

    def process_data(self):
        with open(self.path, 'r') as csvfile:
            datareader = csv.reader(csvfile)
            next(datareader, None)
            for query, positive in datareader:
                yield InputExample(texts=[query, positive], label=1)

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.process_data()


# def extract_data():
#     print('Extracting Data...')
#     data_files = {"train": ["data/sample_val.csv"]}
#     my_iterable_dataset = load_dataset("csv", data_files=data_files, split="train", streaming=True)
#     print(my_iterable_dataset)
#     print(next(iter(my_iterable_dataset)))
#     i = 0
#     # for example in my_iterable_dataset:  # this reads the CSV file progressively as you iterate over the dataset
#     #     print(example)
#     #     i += 1
#     #     if i == 1000:
#     #         break
#
#     # df = pd.read_csv('data/train_data_10M.csv', chunksize=10000000)
#     # df = next(df)
#     # train_data, val_data, test_data = convert_2_InputExample_split(df)
#
#     # with open('data/final_data_10M_InputExample.pkl', 'rb') as f:
#     #     data = pickle.load(f)
#     # n_examples = len(data)
#     # train_data = data[:int(0.8 * n_examples)]
#     # val_data = data[int(0.8 * n_examples):int(0.9 * n_examples)]
#     # test_data = data[int(0.9 * n_examples):]
#
#     # return train_data, val_data, test_data


if __name__ == "__main__":
    s = time.time()
    # extract_data()
    # obj = my_dataset()
    # d = DataLoader(obj)
    # for x in d:
    #     print(x)
    print('Time Taken=', time.time() - s)
