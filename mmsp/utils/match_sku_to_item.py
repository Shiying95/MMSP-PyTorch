# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:25:53 2021

@author: nishiying1
"""

import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
from glob import glob
from PIL import Image
from PIL import ImageChops

# from sklearn.preprocessing import LabelEncoder


def match_sku_to_item(root, verbose=False):
    title = pd.read_csv(f'{root}/new_title.csv', sep='\t')
    # attr = pd.read_csv(f'{root}/new_attr.csv')
    # title = pd.merge(title, attr)
    
    # 找出image相同的sku
    paths = glob(f'{root}/new_picture/*.jpg')
    images = [{'path': path, 'size': os.path.getsize(path)} for path in paths]
    sorted_images = sorted(images, key=lambda x: x['size'])
    
    skus = []
    items = []
    
    item_encoding = 0
    this_path = ''
    last_path = ''
    
    skus.append(os.path.split(sorted_images[0]['path'])[-1][:-4])
    items.append(item_encoding)
    
    for i in tqdm(range(1, len(sorted_images))):
    # for image in sorted_images[:10]:
        this_path = sorted_images[i]['path']
        last_path = sorted_images[i-1]['path']
        
        this_image = Image.open(this_path)
        last_image = Image.open(last_path)

        if this_image != last_image:
            item_encoding += 1
            if verbose:
                print(f'{this_path} and {last_path} are not the same')
        
        sku = os.path.split(this_path)[-1][:-4]
        skus.append(sku)
        items.append(item_encoding)
    
    sku_item_match = pd.DataFrame({'item_sku_id': skus, 'image_item_id': items})
    sku_item_match['item_sku_id'] = sku_item_match['item_sku_id'].astype(np.int64)
    
    title = title.merge(sku_item_match)
    
    # 将名字相同且图片相同的sku作为同一个item
    title['item'] = title['item_name'].str.cat(title['image_item_id'].astype(str))
    
    # 重新编码item
    encoding_map = {}
    
    def encoding(x, encoding_map):
        if x['item'] not in encoding_map:
            item_id = len(encoding_map)
            encoding_map[x['item']] = item_id
        else:
            item_id = encoding_map[x['item']]
        
        return item_id
    
    title['item_id'] = title.apply(encoding, encoding_map=encoding_map, axis=1)
    print(f'match {len(title["item_sku_id"].unique())} skus to {len(title["item_id"].unique())} items')
    
    title[['item_sku_id', 'item_id']].to_csv(f'{root}/item_sku_match.csv', index=False, header=True)
    print(f'save file as {root}/item_sku_match.csv')
    
#%% 划分train/val/test，导出为文件
def split_by_item(root):
    
    df = pd.read_csv(f'{root}/item_sku_match.csv', header=0)
    items = df[['item_id']].drop_duplicates()
    train_items = items.sample(frac=0.7, random_state=17)['item_id'].values.tolist()
    rest = items[~items['item_id'].isin(train_items)]
    val_items = rest.sample(frac=0.5, random_state=17)['item_id'].values.tolist()
    test = rest[~rest['item_id'].isin(val_items)]
    test_items = test['item_id'].values.tolist()
    
    df.loc[df['item_id'].isin(train_items), 'split'] = 'train'
    df.loc[df['item_id'].isin(val_items), 'split'] = 'val'
    df.loc[df['item_id'].isin(test_items), 'split'] = 'test'
    
    df.groupby('split')['item_sku_id'].count()
    
    df.to_csv(f'{root}/split_by_item.csv', index=False, header=True)
    print(f'Saving to {root}/split_by_item.csv')

if __name__ == '__main__':
    root = '../MMSP/data/jd_sleepwear'
    match_sku_to_item(root)


#%% test file
    def get_split_by_file(filepath):
        skus = pd.read_csv(filepath, header=0, dtype={'item_sku_id': np.int64})
        splits = {}
        for split in skus['split'].unique():
            splits[split] = skus[skus['split']==split]['item_sku_id'].values.tolist()
        
        return splits
    
    a = get_split_by_file(filepath='split_by_item.csv')
