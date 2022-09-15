# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:27:27 2021

@author: nishiying1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from mmsp.modules.dist_utils import all_gather


def mask_mse_loss(preds, labels, mask):
    total = mask.sum()  # 有效数据点总数
    mask_preds = preds.mul(mask)  # 无数据部分置零
    mask_labels = labels.mul(mask)  # 无数据部分置零

    loss_sum = F.mse_loss(mask_preds, mask_labels, reduction='sum')  # 对各数据点loss先求和，而非均值
    loss = loss_sum / total  # 除以有效数据点数，得到真实loss
    return loss


def compute_tit(pl_module, batch):
    ret = {}
    prec = pl_module.hparams.config['precision']
    dtype = getattr(torch, f'float{prec}')
    text_image_embed = batch["cls_feats"]
    tit_attrs = torch.tensor(batch['tit_attrs']).to(pl_module.device).type(torch.int64)  # (None, n_tit_cols, maxlen)
    info_attrs = torch.tensor(batch['info_attrs']).to(pl_module.device).type(torch.int64)  # (None, n_info_cols, maxlen)
    
    phase = "train" if pl_module.training else "val"

    # pl_module.output_res['text_image_embed'].append(text_image_embed.clone())
    # pl_module.output_res['item_sku_id'].append(info_attrs.clone())

    for i, col in enumerate(pl_module.tit_cols):
        name = f'{col}_classifier'
        logits = getattr(pl_module, name)(text_image_embed)  # (None, class_token_size) -> (None, n_cls)
        labels = tit_attrs[:, i, 0]  # (None, ) 无需进行one-hot encoding
        loss = F.cross_entropy(input=logits, target=labels, ignore_index=0) * pl_module.tit_loss_factor

        ret.update({
            f"{name}_loss": loss,
            f"{name}_logits": logits,
            f"{name}_labels": labels,
        })

        
        loss = getattr(pl_module, f"{phase}_{name}_loss")(ret[f"{name}_loss"])
        acc = getattr(pl_module, f"{phase}_{name}_accuracy")(ret[f"{name}_logits"], ret[f"{name}_labels"])

        # 仅为train记录每个batch的loss/accuracy，val/test没有batch值
        if phase == 'train':
            pl_module.log(f"batch_tit/{phase}/{name}_loss", loss)
            pl_module.log(f"batch_tit/{phase}/{name}_accuracy", acc)

    return ret


def compute_sp(pl_module, batch):
    ret = {}
    prec = pl_module.hparams.config['precision']
    dtype = getattr(torch, f'float{prec}')

    info_attrs = torch.tensor(batch['info_attrs']).to(pl_module.device).type(torch.int64)  # (None, n_info_cols, maxlen)
    pl_module.output_res['item_sku_id'].append(info_attrs[:, 0].clone())
    pl_module.output_res['item_id'].append(info_attrs[:, 1].clone())

    phase = "train" if pl_module.training else "val"

    if pl_module.embed_dim:
        non_cate_attrs= torch.tensor(batch['non_cate_attrs']).to(pl_module.device).type(dtype)
        cate_attrs = torch.tensor(batch['cate_attrs']).to(pl_module.device).type(torch.int64)

    labels = torch.tensor(batch['label_attrs']).to(pl_module.device).type(dtype)
    # region = torch.tensor(batch['region_attrs']).to(pl_module.device).type(torch.int64)
    # dt = torch.tensor(batch['dt_attrs']).to(pl_module.device).type(torch.int64)
    maxlen = labels.shape[-1]  # the maxlen of datapoints for one item
    split_dim = 1  # 每一维都是一个特征
    
    if pl_module.class_token_size:
        # 直接把text_image_embed分割开作为features，减少降维损失
        text_image_embed = batch["cls_feats"].unsqueeze(1)  # (None, 1, class_token_size)
        text_image_embed_rep = text_image_embed.repeat_interleave(maxlen, dim=1)  # (None, maxlen, class_token_size)
        text_image_embed_list = list(text_image_embed_rep.split(split_dim, dim=-1))  # n * (None, maxlen, embed_dim)

    # deepFM
    first_order_embed_list = list()
    if pl_module.class_token_size and pl_module.with_ti_in_fm:
        # first_order_embed_list.extend(text_image_embed_list)
        for idx, embed in enumerate(text_image_embed_list):
            res = pl_module.first_order_tis[idx](embed)
            first_order_embed_list.append(res)

    if pl_module.embed_dim:
        for i, linear in enumerate(pl_module.first_order_linears):
            non_cate_attr = non_cate_attrs[:, i, :].type(dtype).unsqueeze(-1)  # (None, maxlen) -> (None, maxlen, 1)
            vi = linear(non_cate_attr)  # (None, maxlen, embed_dim)
            xi = torch.ones_like(vi[:, :, 0:1])  # (None, maxlen, 1)  # xi表示的是第i个特征是否有效；我们只考虑了有效的特征，所以都是1
            res = vi * xi  # (None, maxlen, embed_dim)
            first_order_embed_list.append(res)
        
        for i, embed in enumerate(pl_module.first_order_embeddings):
            cate_attr = cate_attrs[:, i, :].type(torch.int64)  # (None, maxlen), Linear和Embedding的输入维不同 
            vi = embed(cate_attr)  # (None, maxlen, embed_dim)
            xi = torch.ones_like(vi[:, :, 0:1])  # (None, maxlen, 1)
            res = vi * xi  # (None, maxlen, embed_dim)
            first_order_embed_list.append(res)
    
    first_order = torch.cat(first_order_embed_list, dim=-1)  # (None, maxlen, total_embed_dim)
    if pl_module.with_mlp:
        first_order = pl_module.mlp(first_order)
        
    prediction = torch.sum(first_order, dim=-1)  # sum along the embed_dim-dimension (None, maxlen, embed_dim) -> (None, maxlen)
    pl_module.output_res[f'{phase}_first_order'].append(prediction.clone())
    # print('prediction', prediction.shape)

    if pl_module.with_second_order:
        second_order_embed_list = list()
        if pl_module.class_token_size and pl_module.with_ti_in_fm:
            # second_order_embed_list.extend(text_image_embed_list)
            for idx, embed in enumerate(text_image_embed_list):
                res = pl_module.second_order_tis[idx](embed)
                second_order_embed_list.append(res)

        if pl_module.embed_dim:
            for i, linear in enumerate(pl_module.second_order_linears):
                non_cate_attr = non_cate_attrs[:, i, :].type(dtype).unsqueeze(-1)  # (None, maxlen, 1)
                vi = linear(non_cate_attr)  # (None, maxlen, embed_dim)
                xi = torch.ones_like(vi[:, :, 0:1])  # (None, maxlen, 1)
                res = vi * xi  # (None, maxlen, embed_dim)
                second_order_embed_list.append(res)
            
            for i, embed in enumerate(pl_module.second_order_embeddings):
                cate_attr = cate_attrs[:, i, :].type(torch.int64)  # (None, maxlen), Linear和Embedding的输入维不同 
                vi = embed(cate_attr)  # (None, maxlen, embed_dim)
                xi = torch.ones_like(vi[:, :, 0:1])  # (None, maxlen, 1)
                res = vi * xi  # (None, maxlen, embed_dim)
                second_order_embed_list.append(res)

        # sum([n * (None, maxlen, embed_dim)]) -> (None, maxlen, embed_dim)
        second_order_embed_sum = sum(second_order_embed_list)  # (None, maxlen, embed_dim)
        second_order_embed_sum_square = second_order_embed_sum * second_order_embed_sum  # (None, maxlen, embed_dim) 
        second_order_embed_square = [res * res for res in second_order_embed_list]
        second_order_embed_square_sum = sum(second_order_embed_square)  # (None, maxlen, embed_dim)
        second_order = 0.5 * (second_order_embed_sum_square - second_order_embed_square_sum)  # (None, maxlen, embed_dim)
        prediction += torch.sum(second_order, dim=-1)

        pl_module.output_res[f'{phase}_second_order'].append(prediction.clone())

    # 如果text-image信息，不参与fm，仅参与deep部分
    if pl_module.class_token_size and (not pl_module.with_ti_in_fm):
        second_order_embed_list.extend(text_image_embed_list)

    # deep nn part
    if pl_module.with_deep:

        deep_embed_list = list()

        if pl_module.second_deep:
            if pl_module.class_token_size:
                deep_embed_list.append(sum(second_order_embed_list[:pl_module.class_token_size]))
                deep_embed_list.extend(second_order_embed_list[pl_module.class_token_size:])
            else:
                deep_embed_list.extend(second_order_embed_list)
        else:
            if pl_module.class_token_size and pl_module.with_ti_in_fm:
                res = pl_module.deep_tis(torch.cat(text_image_embed_list, dim=-1))
                deep_embed_list.append(res)
                # for idx, embed in enumerate(text_image_embed_list):
                #     res = pl_module.deep_tis[idx](embed)
                #     deep_embed_list.append(res)

            if pl_module.embed_dim:
                for i, linear in enumerate(pl_module.deep_linears):
                    non_cate_attr = non_cate_attrs[:, i, :].type(dtype).unsqueeze(-1)  # (None, maxlen, 1)
                    vi = linear(non_cate_attr)  # (None, maxlen, embed_dim)
                    xi = torch.ones_like(vi[:, :, 0:1])  # (None, maxlen, 1)
                    res = vi * xi  # (None, maxlen, embed_dim)
                    deep_embed_list.append(res)
                
                for i, embed in enumerate(pl_module.deep_embeddings):
                    cate_attr = cate_attrs[:, i, :].type(torch.int64)  # (None, maxlen), Linear和Embedding的输入维不同 
                    vi = embed(cate_attr)  # (None, maxlen, embed_dim)
                    xi = torch.ones_like(vi[:, :, 0:1])  # (None, maxlen, 1)
                    res = vi * xi  # (None, maxlen, embed_dim)
                    deep_embed_list.append(res)
    
        deepouts = list()
        deepouts.append(torch.cat(deep_embed_list, axis=-1))  # (None, maxlen, sum(embed_dim))
        # print('deepout', deepouts[-1].shape)
        for i, dnn in enumerate(pl_module.dnn_layers):
            # temp = torch.sum(deepouts[-1], dim=-1)
            # pl_module.output_res[f'{phase}_deep_{i}'].append(temp.clone())
            deepouts.append(dnn(deepouts[-1]))
            
            
        prediction += torch.sum(deepouts[-1], dim=-1)

        pl_module.output_res[f'{phase}_deep'].append(prediction.clone())


    prediction.unsqueeze_(dim=1)  # (None, 1, maxlen)
    
    # prediction = torch.where(
    #     prediction < 0.0,
    #     torch.tensor(0, device=pl_module.device).type(dtype),
    #     prediction)  # 将结果小于0的部分置0    

    mask = torch.where(
        labels==-1,
        torch.tensor(0, device=pl_module.device).type(dtype), 
        torch.tensor(1, device=pl_module.device).type(dtype),
        )

    if pl_module.sp_loss == 'l1_loss':
        sp_loss = F.l1_loss(prediction, labels)
    elif pl_module.sp_loss == 'mse_loss':
        sp_loss = F.mse_loss(prediction, labels)
    else:
        sp_loss = mask_mse_loss(prediction, labels, mask)
    
    sp_ret = {
        "sp_loss": sp_loss,  # 此处loss计入本step的总loss中，根据此进行梯度下降。
        "preds": prediction,
        'labels': labels,
        'mask': mask,
        # 'region': region,
        # 'dt': dt,
    }

    ret.update(sp_ret)
    
    pl_module.output_res[f'{phase}_preds'].append(sp_ret["preds"])
    pl_module.output_res[f'{phase}_labels'].append(sp_ret["labels"])
    
    
    # Metric函数在gadgets.my_metrics中定义。
    # 如果在batch内调用，则传入单step数据，update状态，并自动compute()当前batch内值。
    # 当epoch结束时，调用Metrics.compute()计算最终值。
    sp_loss = getattr(pl_module, f"{phase}_sp_loss")(ret["sp_loss"])
    sp_wmape = getattr(pl_module, f"{phase}_sp_wmape")(ret["preds"], ret["labels"], ret["mask"])
    sp_acc = getattr(pl_module, f"{phase}_sp_acc")(ret["preds"], ret["labels"], ret["mask"])
    sp_mae = getattr(pl_module, f"{phase}_sp_mae")(ret["preds"], ret["labels"], ret["mask"])
    # sp_wmape_region_dt = getattr(pl_module, f"{phase}_sp_wmape_region_dt")(ret["preds"], ret["labels"], ret["mask"], ret["region"], ret["dt"])
    # sp_wmape_region = getattr(pl_module, f"{phase}_sp_wmape_region")(ret["preds"], ret["labels"], ret["mask"], ret["region"])
    sp_wmape_all = getattr(pl_module, f"{phase}_sp_wmape_all")(ret["preds"], ret["labels"], ret["mask"])
    

    # 仅log训练阶段的batch值
    if phase == "train":
        pl_module.log(f"batch_sp/{phase}/loss", sp_loss)
        pl_module.log(f"batch_sp/{phase}/wmape", sp_wmape)
        pl_module.log(f"batch_sp/{phase}/acc", sp_acc)
        
    return ret


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)

    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()