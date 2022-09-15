import random
import torch
import io
import pyarrow as pa
import os
import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from PIL import Image
from mmsp.transforms import (
    keys_to_transforms,
     )

class LightDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        attr_col_file: str,
        transform_keys: list,
        image_size: int,
        max_text_len: int,
        tokenizer: object,
        word_embedding: bool,
        max_image_num: int,
        split: str):

        self.root = root
        self.split = split
        self.attr_col_file = attr_col_file
        self.max_text_len = max_text_len
        self.word_embedding = word_embedding
        self.tokenizer = tokenizer
        self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.max_image_num = max_image_num


        with open(f'{root}/{attr_col_file}') as fp:
            self.attr_cols = json.load(fp)

        if split == 'train':
            file = 'jd_train'
        elif split == 'val':
            file = 'jd_val'
        elif split == 'test':
            file = 'jd_test'

        self.data = pa.ipc.RecordBatchFileReader(
                pa.memory_map(f"{root}/{file}.arrow", "r")
            ).read_all()

        self.valid_attr_cols = list()

        for attr in self.attr_cols.keys():
            if self.attr_cols[attr]:  # 如果attr_cols中没有定义，则忽略之；如果在table中没出现，亦忽略之；
                if isinstance(self.attr_cols[attr], list):
                    cols = self.attr_cols[attr]
                elif isinstance(self.attr_cols[attr], dict):
                    cols = self.attr_cols[attr].keys()

                if set(cols).issubset(set(self.data.column_names)):
                    setattr(self, attr, self.data.select(cols).to_pandas())
                    self.valid_attr_cols.append(attr)
                else:
                    print(f'Error in {attr}: not all cols in {cols} found in the arrow data.')

        for key in ['text_ids', 'text_masks']:
            seq = pad_sequences(
                np.array(self.data[key]), 
                maxlen=self.max_text_len, 
                dtype='int64', 
                padding='post', 
                truncating='pre', 
                value=0,
                )
            setattr(self, key, seq)


    def get_text_attrs(self, idx):

        if self.word_embedding:
            text_ids = self.text_ids[idx]
            text_masks = self.text_masks[idx]
            return {'text_ids': text_ids, 'text_masks': text_masks}
        else:          
            # origin encoding
            text = self.data['item_name'][idx].as_py()
            encoding = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_text_len,
                    return_special_tokens_mask=True,
                    )
            text_ids = np.array(encoding["input_ids"])
            text_masks = np.array(encoding["attention_mask"])
            return {'text_ids': text_ids, 'text_masks': text_masks}
        

    def get_image_attrs(self, idx):

        image_bytes = io.BytesIO(self.data['image'][idx].as_py())
        image_bytes.seek(0)
        image = Image.open(image_bytes).convert("RGB")
        for tr in self.transforms:
            image = tr(image)

        return {
            'image': image.numpy()
        }

        '''
        # 多图模式
        res = []

        img_list = self.data['image'][idx].as_py()
        for img in img_list:
            image_bytes = io.BytesIO(img)
            image_bytes.seek(0)
            image = Image.open(image_bytes).convert("RGB")
            for tr in self.transforms:
                image = tr(image)
            res.append(image.numpy())

        # padding
        image_mask = [1 for i in range(len(res))]

        if len(res) >= self.max_image_num:
            res = res[:self.max_image_num]
            image_mask = image_mask[:self.max_image_num]

        else:
            n_paddings = self.max_image_num - len(res)
            padding_image = np.zeros_like(res[0])
            res += n_paddings * [padding_image]
            image_mask += n_paddings * [0]

        return {
            "image": np.array(res),
            "image_mask": image_mask,
        }
        '''

    def __getitem__(self, idx):
        res = dict()
        try:
            for attr in self.valid_attr_cols:
                res[attr[:-4]+'attrs'] = np.array(getattr(self, attr).iloc[idx, :].values.tolist())
            res.update(self.get_image_attrs(idx))
            res.update(self.get_text_attrs(idx))

        except Exception as e:
            print(f'Error in data {idx} when getting attr cols: {e}')
        
        return res

    def __len__(self):
        return len(self.data)

    def collate(self, batch):
        dict_batch = {}
        batch_size = len(batch)

        # 将list of dict形式转化为dict of list(np.array)形式
        if batch_size:
            keys = batch[0].keys()
            dict_batch = {k: np.array([dic[k] if k in dic else None for dic in batch]) for k in keys}

        return dict_batch