import torch
import torch.nn as nn
import pytorch_lightning as pl
import json
import numpy as np
import os
from collections import defaultdict
from glob import glob
from functools import partial

import mmsp.modules.vision_transformer as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from mmsp.modules import heads, objectives, mmsp_utils


class MMSP(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()  # 将模型参数保存到hparams中
        self.class_token_size = config['class_token_size']
        self.metrics = config['metrics']
        self.hp_metric = config['hp_metric']
        self.sp_loss = config['sp_loss']
        self.external_ti_embedding = config['external_ti_embedding']
        self.second_deep = config['second_deep']


        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        with open(f'{config["workspace_dir"]}/{config["data_root"]}/{config["attr_col_file"]}') as fp:
            attr_config = json.load(fp)

        if self.class_token_size:
            if not self.external_ti_embedding:

                self.text_embeddings = BertEmbeddings(bert_config)  # 直接用transformers包里的Bert Embedding来为文本建模
                self.text_embeddings.apply(objectives.init_weights)  # 加载预训练初始权重
                
                # 如果有没有定义load_path，则根据参数pretrained选择是否自动加载vit默认预训练模型
                if self.hparams.config["load_path"] == "":
                    if self.hparams.config["pretrained"]:
                        self.transformer = getattr(vit, self.hparams.config["vit"])(
                            pretrained=True, config=self.hparams.config
                        )
                    else:
                        self.transformer = getattr(vit, self.hparams.config["vit"])(
                            pretrained=False, config=self.hparams.config
                        )
                # 如果定义了load_path，则不管是否设置pretrained，均不加载预训练权重，而是在之后读取ckpt中的权重
                else:
                    self.transformer = getattr(vit, self.hparams.config["vit"])(
                        pretrained=False, config=self.hparams.config
                    )
            else:
                self.dense_text_embedding =  nn.Linear(config["ti_embedding_size"], config["hidden_size"])
                self.dense_img_embedding =  nn.Linear(config["ti_embedding_size"], config["hidden_size"])
                self.dense_text_embedding.apply(objectives.init_weights)
                self.dense_img_embedding.apply(objectives.init_weights)

                depth = config["depth"]
                dpr = [
                    x.item() for x in torch.linspace(0, drop_path_rate, depth)
                ]  # stochastic depth decay rule
                self.blocks = nn.ModuleList(
                    [
                        vit.Block(
                            dim=config["hidden_size"],
                            num_heads=config["num_heads"],
                            mlp_ratio=config["mlp_ratio"],
                            drop=config["drop_rate"],                            
                        )
                        for i in range(depth)
                    ]
                )
                self.norm = partial(nn.LayerNorm, eps=1e-6)(config["hidden_size"])

            
            self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])  # input_dim=2, output_dim=hidden_size
            self.token_type_embeddings.apply(objectives.init_weights)

            # self.pooler = heads.Pooler(config["hidden_size"], config["class_token_size"])
            self.pooler = heads.MeanPooler(config["hidden_size"], config["class_token_size"])
            self.pooler.apply(objectives.init_weights)
        else:
            config["loss_names"]["tit"] = 0
        

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")  # load预训练好的参数
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]


        # 离散变量需要embedding
        self.cate_cols = attr_config['cate_cols']
        self.non_cate_cols = attr_config['non_cate_cols']
        
        self.n_cate_cols = len(self.cate_cols)
        self.n_non_cate_cols = len(self.non_cate_cols)
        
        if config["loss_names"]["tit"] > 0:
            self.tit_cols = attr_config['tit_cols']
            self.tit_loss_factor = self.hparams.config['tit_loss_factor']

            for col, n_cls in self.tit_cols.items():
                tit_classifier = nn.Sequential(
                    nn.Linear(self.class_token_size, self.class_token_size),
                    nn.LayerNorm(self.class_token_size),
                    nn.GELU(),
                    nn.Linear(self.class_token_size, n_cls),
                    # nn.Softmax(dim=-1),  # 内置的cross_entropy包含了softmax部分
                )

                setattr(self, f'{col}_classifier', tit_classifier)
                getattr(self, f'{col}_classifier').apply(objectives.init_weights)            

        if self.hparams.config["loss_names"]["sp"] > 0:

            n_fields = self.n_non_cate_cols + self.n_cate_cols # 1 for text-image embed
            fm_embed_dim = config['fm_embed_dim']
            fm_hidden_dims = config['fm_hidden_dims']
            fm_dropouts = config['fm_dropouts']
            total_dims = (n_fields + 1) * fm_embed_dim
            
            self.embed_dim = fm_embed_dim
            self.with_deep = config['with_deep']
            if self.with_deep:
                self.with_second_order = True  # deep需要second order作为输入

            else:
                self.with_second_order = config['with_second_order']
            self.with_mlp = config['with_mlp']
            self.with_ti_in_fm = config['with_ti_in_fm']

            # 如果考虑结构化数据，才需要first(second)_order_linears(embeddings)
            if self.embed_dim:
                # DeepFM
                # first order
                self.first_order_linears = nn.ModuleList()
                self.first_order_embeddings = nn.ModuleList()

                for col in self.non_cate_cols:
                    first_order_linear = nn.Linear(1, 1)
                    first_order_linear.apply(objectives.init_weights)
                    self.first_order_linears.append(first_order_linear)      

                for col, n_features in self.cate_cols.items():                
                    # first_order_embedding = nn.Embedding(n_features, fm_embed_dim, padding_idx=0)
                    first_order_embedding = nn.Embedding(n_features, 1)
                    first_order_embedding.apply(objectives.init_weights)
                    self.first_order_embeddings.append(first_order_embedding)

                if self.class_token_size:
                    self.first_order_tis = nn.ModuleList()
                    for i in range(self.class_token_size):
                        first_order_ti = nn.Linear(1, 1)
                        first_order_ti.apply(objectives.init_weights)
                        self.first_order_tis.append(first_order_ti)

                if self.with_mlp:
                    self.mlp = nn.Sequential(
                        nn.Linear(total_dims, 32),
                        # nn.LayerNorm(32),
                        nn.GELU(),
                        )
                    self.mlp.apply(objectives.init_weights)

                # second order
                if self.with_second_order:
                    self.second_order_linears = nn.ModuleList()
                    self.second_order_embeddings = nn.ModuleList()

                    for col in self.non_cate_cols:
                        second_order_linear = nn.Linear(1, fm_embed_dim, bias=False)
                        second_order_linear.apply(objectives.init_weights)
                        self.second_order_linears.append(second_order_linear)

                    for col, n_features in self.cate_cols.items():
                        # second_order_embedding = nn.Embedding(n_features, fm_embed_dim, padding_idx=0)
                        second_order_embedding = nn.Embedding(n_features, fm_embed_dim)
                        second_order_embedding.apply(objectives.init_weights)
                        self.second_order_embeddings.append(second_order_embedding)

                    if self.class_token_size:
                        self.second_order_tis = nn.ModuleList()
                        for i in range(self.class_token_size):
                            second_order_ti = nn.Linear(1, fm_embed_dim, bias=False)
                            second_order_ti.apply(objectives.init_weights)
                            self.second_order_tis.append(second_order_ti)

            # deep nn
            if self.with_deep:

                if not self.second_deep:

                    self.deep_linears = nn.ModuleList()
                    self.deep_embeddings = nn.ModuleList()

                    for col in self.non_cate_cols:
                        deep_linear = nn.Linear(1, fm_embed_dim, bias=False)
                        deep_linear.apply(objectives.init_weights)
                        self.deep_linears.append(deep_linear)

                    for col, n_features in self.cate_cols.items():
                        # deep_embedding = nn.Embedding(n_features, fm_embed_dim, padding_idx=0)
                        deep_embedding = nn.Embedding(n_features, fm_embed_dim)
                        deep_embedding.apply(objectives.init_weights)
                        self.deep_embeddings.append(deep_embedding)

                    if self.class_token_size:
                        self.deep_tis = nn.Linear(self.class_token_size, fm_embed_dim, bias=False)
                        # self.deep_tis = nn.ModuleList()
                        # for i in range(self.class_token_size):
                        #     deep_ti = nn.Linear(1, fm_embed_dim, bias=False)
                        #     deep_ti.apply(objectives.init_weights)
                        #     self.deep_tis.append(deep_ti)
                
                if not self.class_token_size:
                    total_dims = n_fields * fm_embed_dim



                fm_dnn_dims = [total_dims] + fm_hidden_dims  # dims of dnn layers
                # print('total_dims', total_dims)
                self.dnn_layers = nn.ModuleList()
                for i in range(len(fm_dnn_dims)-1):
                    # print(f'dnn layer {i}: {fm_dnn_dims[i]}, {fm_dnn_dims[i+1]}')
                    dnn_layer = nn.Sequential(
                        nn.Linear(fm_dnn_dims[i], fm_dnn_dims[i+1]),  # (None, maxlen, dim)
                        # nn.GELU(),
                        # PermuteLayer(dim=(0, 2, 1)),  # (None, dim, maxlen)
                        # nn.BatchNorm1d(fm_dnn_dims[i+1]),  # BatchNorm1d on dim
                        # PermuteLayer(dim=(0, 2, 1)),  # (None, maxlen, dim)
                        # nn.BatchNorm1d(1),  # BatchNorm1d on dim
                        nn.ReLU(),
                        # nn.LayerNorm(fm_dnn_dims[i+1]),  # TODO
                        nn.Dropout(fm_dropouts[i]),
                        )
                    dnn_layer.apply(objectives.init_weights)
                    self.dnn_layers.append(dnn_layer)

        mmsp_utils.set_metrics(self)  # 为所有的指标赋Metric类
        self.current_tasks = list()

        self.output_res = defaultdict(list)
        self.current_gpu = 0

        # ===================== load downstream (test_only) ======================
        if config["test_only"]:
            if config["test_version"] is not None:
                log_dir = os.path.abspath(f'{config["workspace_dir"]}/{config["log_dir"]}')
                load_dir = f'{log_dir}/{config["exp_name"]}_seed{config["seed"]}/version_{config["test_version"]}/checkpoints'
                # print(load_dir)
                # ckpt_path = glob(f'{load_dir}/epoch=*.ckpt')[0]
                ckpt_path = glob(f'{load_dir}/last.ckpt')[0]
                print(f'load ckpt from {ckpt_path}')
                ckpt = torch.load(ckpt_path, map_location="cpu")
                state_dict = ckpt["state_dict"]
                self.load_state_dict(state_dict, strict=False)
            elif config["load_path"] != "":
                ckpt = torch.load(config["load_path"], map_location="cpu")
                state_dict = ckpt["state_dict"]
                self.load_state_dict(state_dict, strict=False)
            else:
                self.print('Do not load any ckpt...')
    
    def infer_image_text(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        
        prec = self.hparams.config['precision']
        dtype = getattr(torch, f'float{prec}')

        if self.hparams.config['external_ti_embedding']:


            text_embeds = torch.tensor(batch['text_attrs']).to(self.device).type(dtype)
            image_embeds = torch.tensor(batch['img_attrs']).to(self.device).type(dtype)

            text_embeds = self.dense_text_embedding(text_embeds)
            image_embeds = self.dense_img_embedding(image_embeds)

            text_masks = torch.ones_like(text_embeds[:, :, 0]).to(self.device).type(torch.int64)
            image_masks = torch.ones_like(image_embeds[:, :, 0]).to(self.device).type(torch.int64)

            # print('text embeds:', text_embeds.shape)
            # print('text masks:', text_masks.shape)

            text_embeds, image_embeds = (
                text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
                image_embeds + self.token_type_embeddings(
                    torch.full_like(image_masks, image_token_type_idx)
                ),
            )

            co_embeds = torch.cat([text_embeds, image_embeds], dim=1)  # 直接拼接
            co_masks = torch.cat([text_masks, image_masks], dim=1)  # 直接拼接

            # print('co embeds:', co_embeds.shape)

            x = co_embeds

            for i, blk in enumerate(self.blocks):  # 依次经过多个block
                x, _attn = blk(x, mask=co_masks)

            x = self.norm(x)

            # x = torch.cat([text_embeds, image_embeds], dim=-1)  # 拼接

            # print('x:', x.shape)

            cls_feats = self.pooler(x)  # 进行pool

            ret = {
                "text_feats": text_embeds,
                "image_feats": image_embeds,
                "cls_feats": cls_feats,
            }

        else:
            imgkey = "image"
            text_ids = torch.tensor(batch[f"text_ids"]).to(self.device).type(torch.int64)
            text_masks = torch.tensor(batch[f"text_masks"]).to(self.device).type(torch.int64)
            text_embeds = self.text_embeddings(text_ids)

            if image_embeds is None and image_masks is None:
                img = torch.tensor(batch[imgkey]).to(self.device).type(dtype)

                (
                    image_embeds,
                    image_masks,
                    patch_index,
                    image_labels,
                ) = self.transformer.visual_embed(
                    img,
                    max_image_len=self.hparams.config["max_image_len"],
                    mask_it=mask_image,
                )
                
            else:
                patch_index, image_labels = (
                    None,
                    None,
                )

            text_embeds, image_embeds = (
                text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
                image_embeds
                + self.token_type_embeddings(
                    torch.full_like(image_masks, image_token_type_idx)
                ),
            )

            image_embeds = image_embeds[:, 1:, :]
            image_masks = image_masks[:, 1:]

            # print('text embeds:', text_embeds.shape)  # text embedding没有cls token
            # print('img embeds:', image_embeds.shape)  # image embedding有cls token


            co_embeds = torch.cat([text_embeds, image_embeds], dim=1)  # 直接拼接，去除cls token
            co_masks = torch.cat([text_masks, image_masks], dim=1)  # 直接拼接

            # print('co embeds:', co_embeds.shape)

            x = co_embeds

            

            for i, blk in enumerate(self.transformer.blocks):  # 依次经过多个block
                x, _attn = blk(x, mask=co_masks)

            x = self.transformer.norm(x)

            print('x', x.shape)

            text_feats, image_feats = (  
                x[:, : text_embeds.shape[1]],
                x[:, text_embeds.shape[1] :],
            ) 

            cls_feats = self.pooler(x)  # 进行pool


            ret = {
                "text_feats": text_feats,
                "image_feats": image_feats,
                "cls_feats": cls_feats,
                "raw_cls_feats": x[:, 0],
                "image_labels": image_labels,
                "image_masks": image_masks,
                # "text_labels": text_labels,
                "text_ids": text_ids,
                "text_masks": text_masks,
                "patch_index": patch_index,
            }


        return ret
    

    def forward(self, batch):  # 相当于tf2中call函数中的内容
        ret = dict()
        if self.class_token_size:
            batch.update(self.infer_image_text(batch))  # 将训练好的image-text信息整合到batch中

        # Text-image co-training
        if 'tit' in self.current_tasks:
            ret.update(objectives.compute_tit(self, batch))

        # Sale Prediction
        if "sp" in self.current_tasks:
            ret.update(objectives.compute_sp(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        mmsp_utils.set_task(self)  # 往self.current_tasks中添加我们需要的任务
        output = self(batch)  # self指pl.LightningModule, 它的call函数里写入了.forward()的内容
        total_loss = sum([v for k, v in output.items() if "loss" in k])  # 将所有的loss加起来成为总loss，用于模型梯度下降训练
        self.example_input_array = batch
        return total_loss

    def training_epoch_end(self, outs):
        mmsp_utils.epoch_wrapup(self)  # epoch结束后，cal and reset loss/metrics

    def validation_step(self, batch, batch_idx):
        mmsp_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):

        # 当为fast run模式时，没有log_dir，导致报错
        if not self.hparams.config['fast_dev_run']:
            npz = {}
            for key in self.output_res:
                npz[key] = torch.cat(self.output_res[key]).cpu().numpy()

            os.makedirs(f'{self.trainer.log_dir}/outputs/', exist_ok=True)
            path = f'{self.trainer.log_dir}/outputs/epoch_{self.trainer.current_epoch}_{self.device}.npz'
            np.savez(path, **npz)

        self.output_res.clear()
        mmsp_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        mmsp_utils.set_task(self)
        output = self(batch)
        ret = dict()

        # if self.hparams.config["loss_names"]["vqa"] > 0:
        #     ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]
        print('test epoch')

        # 当为fast run模式时，没有log_dir，导致报错
        if not self.hparams.config['fast_dev_run']:
            npz = {}
            for key in self.output_res:
                npz[key] = torch.cat(self.output_res[key]).cpu().numpy()

            os.makedirs(f'{self.trainer.log_dir}/outputs/', exist_ok=True)
            path = f'{self.trainer.log_dir}/outputs/epoch_{self.trainer.current_epoch}_{self.device}.npz'
            np.savez(path, **npz)

        self.output_res.clear()

        mmsp_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return mmsp_utils.set_schedule(self)


class PermuteLayer(nn.Module):
    def __init__(self, dim, **kwargs):
        super(PermuteLayer, self).__init__(**kwargs)
        self.dim = dim

    def forward(self, x):
        out = x.permute(self.dim)
        return out