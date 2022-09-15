import torch
import os
from pytorch_lightning import LightningDataModule
from mmsp.datasets.light_dataset import LightDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)


CACHE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))+'/cache'
os.environ['TORCH_HOME'] = CACHE_DIR

def get_pretrained_tokenizer(from_pretrained):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            BertTokenizer.from_pretrained(
                from_pretrained, cache_dir=f'{CACHE_DIR}/{from_pretrained}', do_lower_case="uncased" in from_pretrained
            )
        torch.distributed.barrier()
    return BertTokenizer.from_pretrained(
        from_pretrained, cache_dir=f'{CACHE_DIR}/{from_pretrained}', do_lower_case="uncased" in from_pretrained
    )

class LightDataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        super().__init__()
        self.root = f'{_config["workspace_dir"]}/{_config["data_root"]}'
        self.attr_col_file = _config['attr_col_file']
        self.dist = dist
        self.batch_size = _config['per_gpu_batchsize']
        self.num_workers = _config['num_workers']
        self.word_embedding = _config['word_embedding']
        self.max_image_num = _config['max_image_num']

        # for image processing
        self.image_size = _config["image_size"]
        self.train_transform_keys = (
            ["default_train"]
            if len(_config["train_transform_keys"]) == 0
            else _config["train_transform_keys"]
        )
        self.val_transform_keys = (
            ["default_val"]
            if len(_config["val_transform_keys"]) == 0
            else _config["val_transform_keys"]
        )

        # for text processing
        self.max_text_len = _config["max_text_len"]
        if self.word_embedding:
            self.tokenizer = None
        else:
            self.tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
            # self.vocab_size = self.tokenizer.vocab_size


    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        return

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.train_dataset = LightDataset(
            root=self.root, 
            attr_col_file=self.attr_col_file,
            transform_keys=self.train_transform_keys,
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            tokenizer=self.tokenizer,
            word_embedding=self.word_embedding,
            max_image_num=self.max_image_num,
            split='train',
            )

        self.val_dataset = LightDataset(
            root=self.root, 
            attr_col_file=self.attr_col_file,
            transform_keys=self.train_transform_keys,
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            tokenizer=self.tokenizer,
            word_embedding=self.word_embedding,
            max_image_num=self.max_image_num,
            split='val',
            )

        self.test_dataset = LightDataset(
            root=self.root, 
            attr_col_file=self.attr_col_file,
            transform_keys=self.train_transform_keys,
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            tokenizer=self.tokenizer,
            word_embedding=self.word_embedding,
            max_image_num=self.max_image_num,
            split='test',
            )

        if self.dist:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True)
            self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None


    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate,  # merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.
        )
        return loader

    def val_dataloader(self, batch_size=None):
        loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate,
        )
        return loader

    def test_dataloader(self, batch_size=None):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset.collate,
        )
        return loader