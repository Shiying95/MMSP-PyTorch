# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 10:18:49 2021

@author: nishiying1
"""

from collections import defaultdict
import itertools
import json
import os
import re
from ast import literal_eval

import pandas as pd
import numpy as np
import pyarrow as pa
from tqdm import tqdm
from glob import glob
import jieba
from sklearn.preprocessing import LabelEncoder

np.random.seed(17)  # 设置随机种子

def get_split_by_time(attrs, train_end, val_end, test_end):
    '''
    按照时间划分训练、验证、测试集
    '''
    train = attrs[attrs['sale_ord_dt'] <= train_end].values.tolist()
    val = attrs[(attrs['sale_ord_dt'] > train_end) & (attrs['sale_ord_dt'] <= val_end)].values.tolist()
    test = attrs[(attrs['sale_ord_dt'] > val_end) & (attrs['sale_ord_dt'] <= test_end)].values.tolist()
    
    if val:
        return {'train': train, 'val': val, 'test': test}
    else:
        return {'train': train, 'test': test}
    

def get_split_by_frac(attrs, train_frac, val_frac, test_frac):
    '''
    按照sku比例划分训练、验证、测试集
    '''
    skus = attrs['item_sku_id'].drop_duplicates()
    train = skus.sample(frac=train_frac, replace=False)
    rest = skus[~skus.index.isin(train.index)]
    if test_frac:  # 如果设定了test_frac，则继续分
        val = rest.sample(frac=val_frac/(1-train_frac), replace=False)
        test = rest[~rest.index.isin(val.index)]
    else:
        val = rest
        test = pd.Series(dtype=np.int64)
        print('no testset.')
    
    splits = {'train': train.tolist(), 'val': val.tolist(), 'test': test.tolist()}

    return splits

def get_split_by_file(filepath):
    skus = pd.read_csv(filepath, header=0, dtype={'item_sku_id': np.int64})
    splits = {}
    for split in skus['split'].unique():
        splits[split] = skus[skus['split']==split]['item_sku_id'].values.tolist()
    
    return splits


def tokenize(text):

    text = text.upper()  # 全大写
    # 去除异常值
    for char in [' ', '/', '(', ')', '（', '）', '【', '】', '《', '》', '.']:
        text = text.replace(char, '')
    text_seg = jieba.lcut(text)  # 分词
    # 仅保留含有汉字的部分
    res = []
    for word in text_seg:
        if (re.search(u'[\u4e00-\u9fa5]+', word)):
            res.append(word)
    return res


def prepare_text_image(root, arrows_root):
    print('reading text and image data...')
    captions = pd.read_csv(f'{root}/new_title.csv', sep='\t', dtype={'item_sku_id': np.int64},)
    captions = captions.to_dict(orient='records')

    print('preprocessing text data...')

    jieba.load_userdict(f'{root}/jieba_dict.txt')

    vocab = set()
    for record in captions:
        record['words'] = tokenize(record['item_name'])
        vocab = vocab.union(record['words']) 

    print(f'vocabulary: {len(vocab)}')

    encoding = {}
    decoding = {}
    idx = 103
    for word in vocab:
        encoding[word] = idx
        decoding[idx] = word
        idx += 1

    with open(f'{arrows_root}/word_decoding.json', 'w') as fp:
        json.dump(decoding, fp)

    for record in captions:
        record['text_ids'] = [encoding[word] for word in record['words']]
        record['text_ids'].insert(0, 101)  # 初始token
        record['text_ids'].append(102)  # 结束token
        record['text_masks'] = [1 for i in record['text_ids']]
        # record['text'] = ''.join(record['words'])

    for record in captions:
        try:
            with open(f'{root}/new_picture/{record["item_sku_id"]}.jpg', 'rb') as fp:
                binary = fp.read()  # 读取图片
            record['image'] = binary
        except Exception as e:
            print(f'Error in {record["item_sku_id"]}: {e}')
    
    '''
    # 多图模式  # TODO
    for record in captions:
        try:
            pic_paths = glob(f'{root}/multi_picture/{record["item_sku_id"]}/*.jpg')
            pic_list = []
            for pic_path in pic_paths:
                with open(pic_path, 'rb') as fp:
                    binary = fp.read()  # 读取图片
                    pic_list.append(binary)
            record['image'] = pic_list
        except Exception as e:
            print(f'Error in {record["item_sku_id"]}: {e}')
    '''
    
    ti_attr = pd.DataFrame(captions)
    # for col in ['item_name', 'text', 'words']:  # 将文字类型的列去掉，使数据处理更快
    for col in ['text', 'words']:
        if col in ti_attr.columns:
            ti_attr.drop(col, axis=1, inplace=True)

    print(f'read {len(ti_attr)} text-image records')

    return ti_attr


def prepare_ns_attrs(root, arrows_root):
    # get non-sequential attributes
    attr = pd.read_csv(
        f'{root}/new_attr.csv',
        header=0,
        dtype={'item_sku_id': np.int64},
        # index_col=0,
        # sep='\t',
        )
    
    le_attr = attr[['item_sku_id']].copy()
    le_dict = dict()
    cols = set(attr.columns) - {'item_sku_id', 'brand_code'}  # 不需要brand_code
    
    for col in cols:
        le = LabelEncoder()
        le_attr[col] = le.fit_transform(attr[col].astype(str)).astype(np.int64) # 把0空出来
        le_map = le.classes_.tolist()
        
        # 将0值留给mask和nan
        nan = 'nan'
        if nan in le_map:
            idx = le_map.index(nan)
            exchange_value = le_map[0]
            le_map[0] = nan
            le_map[idx] = exchange_value
            
            dummy = -1
            le_attr.loc[le_attr[col]==0, col] = dummy
            le_attr.loc[le_attr[col]==idx, col] = 0
            le_attr.loc[le_attr[col]==dummy, col] = idx
            
        else:
            le_map.insert(0, 'nan')
            le_attr[col] = le_attr[col] + 1  # 因为额外添加0，所以整体平移
        
        le_dict[col] = le_map

    with open(f'{arrows_root}/label_encoding.json', 'w') as fp:
        json.dump(le_dict, fp)
    
    return le_attr


def prepare_attrs(root, arrows_root, days_per_period, n_periods, mode, debug, trunc_sales):
    '''
    读取属性文件，并对原始数据进行清洗，保证字段均为float或int格式，否则在创建.arrow文件时会出错
    '''
    if debug:
        nrows=100000
    else:
        nrows=None

    sales = pd.read_csv(
        f'{root}/new_sale.csv', 
        header=0, 
        dtype={'item_sku_id': np.int64},
        sep='\t',
        nrows=nrows,
        )
    print(f'read {len(sales)} attr records with {len(sales.columns)} cols')
    
    # 转化为时间格式
    for col in ['sale_ord_dt', 'shelves_dt']:
        sales[col] = pd.to_datetime(sales[col])

    # 去除方差特别大的sku
    print(f'trunc series with large std... before: {len(sales)}')
    ratio = sales.groupby('item_sku_id')['sale_qtty'].std() / sales.groupby('item_sku_id')['sale_qtty'].mean() 
    count = sales.groupby('item_sku_id')['sale_qtty'].count()
    invalid_skus = ratio[ratio > 0.7 * np.sqrt(count-1)].index
    sales =  sales[~sales['item_sku_id'].isin(invalid_skus)]
    print(f'after: {len(sales)}')

    
    # 按照指定period长度来统计销量
    attrs = sales.groupby(['item_sku_id', 'sale_ord_dt'])['sale_qtty'].sum().reset_index()
    on_shelf = sales[['item_sku_id', 'shelves_dt']].drop_duplicates()
    attrs = pd.merge(attrs, on_shelf, how='inner')
    
    attrs['on_shelf_days'] = (attrs['sale_ord_dt'] - attrs['shelves_dt']).dt.days  # 从0开始
    attrs['on_shelf_periods'] = attrs['on_shelf_days'] // days_per_period 
    
    # 选取指定时间长度里销量完整的sku
    valid = attrs.groupby(['item_sku_id'])['on_shelf_days'].max().reset_index()
    valid_skus = valid[valid['on_shelf_days'] >= (days_per_period * n_periods - 1)]['item_sku_id'].tolist()
    print(f'remain {len(valid_skus)} valid skus with full history')
    
    attrs = attrs[attrs['item_sku_id'].isin(valid_skus)]
    attrs = attrs[attrs['on_shelf_periods'] < n_periods]  # 仅选取指定个period的数据
    
    sale_group = attrs.groupby(['item_sku_id', 'on_shelf_periods'])
    
    res = pd.DataFrame({
        'sale_qtty': sale_group['sale_qtty'].sum(),
        'sale_ord_dt': sale_group['sale_ord_dt'].min(),
        }).reset_index()
    
    price = sales.drop_duplicates(['item_sku_id', 'jd_prc'])[['item_sku_id', 'jd_prc']]
    res = pd.merge(res, price, on='item_sku_id', how='left')
    
    res['year_no'] = res['sale_ord_dt'].dt.isocalendar().year.astype(np.int64)
    res['month_no'] = res['sale_ord_dt'].dt.month.astype(np.int64)
    res['week_no'] = res['sale_ord_dt'].dt.isocalendar().week.astype(np.int64)
    res['day_no'] = res['sale_ord_dt'].dt.day.astype(np.int64)
    
    print(f'remain skus: {len(res["item_sku_id"].unique())}, records: {len(res)}')
    
    ns_attrs = prepare_ns_attrs(root, arrows_root)
    print(f'read {len(ns_attrs.columns)-1} non-sequential attrs.')
    
    attrs = pd.merge(res, ns_attrs, on='item_sku_id', how='inner')
    print(f'remain {len(attrs)} records after merging non-sequential attrs')


    
    attrs = attrs.sort_values(['item_sku_id', 'on_shelf_periods'])
    
    attrs['sale_ord_dt'] = attrs['sale_ord_dt'].dt.strftime('%Y%m%d')
    attrs['sale_ord_dt'] = attrs['sale_ord_dt'].astype(np.int64)
    attrs['dummy'] = 1
    
    # 列值转换
    # attrs.loc[attrs['is_weekday']==0, 'is_weekday'] = 2  # 将0值留出来给mask
    
    for col in attrs.columns:
        for value in [np.inf, -np.inf]:
            if len(attrs.loc[attrs[col]==value, col]):
                attrs.loc[attrs[col]==value, col] = 1
                print(f'covert {value} in col {col} to 1.')

    sku_item_match = pd.read_csv(
        f'{root}/item_sku_match.csv', 
        header=0, 
        dtype={
            'item_sku_id': np.int64,
            'item_id': np.int64,
            },
        )

    attrs = pd.merge(attrs, sku_item_match, on='item_sku_id', how='inner')

    if trunc_sales:
        item_sales = attrs.groupby(['item_id'])['sale_qtty'].sum().reset_index()
        trunc_value = item_sales['sale_qtty'].mean() + 3 * np.sqrt(item_sales['sale_qtty'].var())
        valid_items = item_sales[item_sales['sale_qtty'] <= trunc_value]['item_id'].values.tolist()
        attrs = attrs[attrs['item_id'].isin(valid_items)]
        print(f'remain {len(attrs)} records after truncating items with sales > {trunc_value:.1f} (mu+3*sigma)')

        
    if 'file' in mode:
        splits = get_split_by_file(f'{root}/split_by_item.csv')
    elif 'frac' in mode:
        splits = get_split_by_frac(attrs, train_frac=0.7, val_frac=0.15, test_frac=0.15)
        with open(f'{root}/splits_by_frac.json', 'w') as fp:
            json.dump(splits, fp)
    elif 'time' in mode:
        splits = get_split_by_time(attrs, train_end=20000000, val_end=20000000, test_end=20000000)
    
    for split, skus in splits.items():
        attrs.loc[attrs['item_sku_id'].isin(skus), 'split'] = split


    # 去除空值行
    rows_before = len(attrs)
    attrs.dropna(axis=0, how='any', inplace=True)
    rows_after = len(attrs)
    if rows_before == rows_after:
        print('no nan data detected.')
    else:
        print(f'drop {rows_before-rows_after} rows with nan value.')
    
    for i in attrs:
        if attrs[i].dtype not in [np.dtype('int64'), np.dtype('int32'), np.dtype('float')]:
            print(f'type invalid: {i}, {attrs[i].dtype}, may cause error when making arrows')

    print(f'columns in structured data: {list(attrs.columns)}')
    
    return attrs


def get_attrs(attrs, maxlen):
    '''
    将属于同一个item_sku_id的datapoints整合到一条记录。    
    '''
    
    maxlen = maxlen
    res = defaultdict(list)

    for sku_id, df in tqdm(attrs.groupby('item_sku_id')):
        res['item_sku_id'].append(sku_id)  # 保持单值，方面之后的merge
        res['item_id'].append(df['item_id'].to_list()[0])  # 保持单值，和item_sku_id一起加入info列
        for col in set(df.columns)-set(['item_sku_id', 'item_id']):
            feature = df[col].to_list()
            if len(feature) != maxlen:  # 判断每条sku下记录数是否正确
                print(f'# of records in {sku_id} != maxlen ({len(feature)} vs. {maxlen}), please check!')
                continue
            else:
                res[col].append(feature)
            
    return pd.DataFrame(res)

    
def make_arrow(
        root, 
        arrows_root, 
        days_per_period=7, 
        n_periods=7,
        mode='split_by_file',
        debug=False,
        with_ti_embedding=False,
        trunc_sales=True,
        ):

    os.makedirs(f'{arrows_root}', exist_ok=True)
    
    # get structured data
    attrs = prepare_attrs(
        root,
        arrows_root,
        days_per_period=days_per_period, 
        n_periods=n_periods,
        mode=mode,
        debug=debug,
        trunc_sales=trunc_sales,
        )

    image_text = prepare_text_image(root, arrows_root)
    if with_ti_embedding:
        try:
            print('preparing ti embedding data...')
            ti_embedding = pd.read_csv(f'{root}/BRIVL_feature.csv')
            for col in ['img_feature', 'text_feature']:
                ti_embedding[col] = ti_embedding.apply(lambda x: literal_eval(x[col]), axis=1)
            print(f'read {len(ti_embedding)} ti embedding data.')
        except Exception as e:
            print(f'Error in reading ti embedding: {e}')
            ti_embedding = None
    else:
        ti_embedding = None
        
    sku_list = dict()  # for sku reduplication check

    for split in attrs['split'].unique():
        dataset = attrs[attrs['split']==split]
        dataset.drop(['split'], axis=1, inplace=True)

        os.makedirs(arrows_root, exist_ok=True)

        print(f'making csv for {split} datasets without text-image information...')
        dataset.to_csv(f'{arrows_root}/{split}.csv', header=True, index=True)
        
        print(f'preparing {split} attrs: {len(dataset)} valid records with maxlen={n_periods}')
        attr = get_attrs(dataset, maxlen=n_periods)

        dataframe = pd.merge(image_text, attr, on='item_sku_id', how='inner')
        if ti_embedding is not None:
            dataframe = pd.merge(dataframe, ti_embedding, on='item_sku_id', how='inner')
        sku_list[split] = set(dataframe['item_sku_id'].to_list())

        print(f'making arrow for {split} datasets: {len(dataframe)} skus')
        table = pa.Table.from_pandas(dataframe)
        
        with pa.OSFile(
            f"{arrows_root}/jd_{split}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
    
    for combination in itertools.combinations(sku_list, 2):
        split1, split2 = combination
        if split1 != split2:
            print(f'intersection of skus in {split1} and {split2}: {len(sku_list[split1] & sku_list[split2])}')
        
    print('make arrow successfully')       


if __name__ == '__main__':
    root = '../../data/jd_t-shirt'
    arrows_root = '../../data/arrows/'
    debug = True
    make_arrow(root, arrows_root, debug=debug)
    