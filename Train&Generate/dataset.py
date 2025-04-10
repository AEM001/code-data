import re
import math
import string
import os
import torch
import pytorch_lightning as pl
import pickle

from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader

from itertools import chain
from transformers import default_data_collator


def prepare_data(path='./data/wikitext2'):
    # 检查本地是否已有数据集
    if os.path.exists(path):
        print(f"从本地加载数据集 {path}...")
        try:
            dataset = load_from_disk(path)
            print(f"成功从本地加载数据集")
            return dataset
        except Exception as e:
            print(f"从本地加载数据集失败: {e}")
            print("尝试从HuggingFace重新下载...")
    
    # 如果本地没有或加载失败，从HuggingFace下载
    print("从HuggingFace下载数据集...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    print(f"成功从HuggingFace加载数据集")
    
    # 保存到本地以便下次使用
    os.makedirs(path, exist_ok=True)
    dataset.save_to_disk(path)
    print(f"数据集已保存到 {path}")
    
    return dataset


def preprocess_datasets(raw_dataset, tokenizer, block_size=64, overwrite_cache=False, preprocessing_num_workers=4):
    column_names = raw_dataset['train'].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if block_size is None:
        block_size = tokenizer.model_max_length

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not overwrite_cache,
        desc="Running tokenizer on dataset",
        keep_in_memory=True,
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
        keep_in_memory=True
    )
    return dataset


class MyDataLoader(pl.LightningDataModule):
    def __init__(self, dataset_name, workers, train_dataset, val_dataset, test_dataset, batch_size):
        super().__init__()
        self.dataset_name = dataset_name
        self.train_dataset, self.val_dataset, self.test_dataset = train_dataset, val_dataset, test_dataset
        self.batch_size = batch_size
        self.num_workers = workers

        # 建议添加pin_memory参数加速GPU传输
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        # 同样修改val和test的dataloader
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

class WikiText2Dataset(Dataset):
    path = './data/wikitext2'
    def __init__(
            self, 
            dataset,
            partition,
            tokenizer=None, 
            max_token_count=512):
        self.setup_tokenizer(tokenizer, max_token_count)
        self.dataset = dataset[partition]

    def setup_tokenizer(self, tokenizer, max_token_count):
        self.tokenizer = tokenizer
        self.max_token_count = max_token_count

    def __len__(self):
        return self.dataset.num_rows

    # 将第二个WikiText2Dataset类定义删除，修改__getitem__方法
    def __getitem__(self, index):
        data_row = self.dataset[index]
        # 添加长度检查和截断
        input_ids = data_row['input_ids'][:self.max_token_count]
        attention_mask = data_row['attention_mask'][:self.max_token_count]
        labels = data_row['labels'][:self.max_token_count]
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }


class GeneratedDataset(Dataset):
    path = './data/wikitext2'

    def __init__(
            self,
            my_file,
            partition,
            tokenizer=None,
            max_token_count=512):
        with open(my_file, 'rb') as f:
            self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_row = self.dataset[index]
        return dict(
            input_ids=data_row['input_ids'],
            attention_mask=data_row['attention_mask'],
            labels=data_row['labels'])
