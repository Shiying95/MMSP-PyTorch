from sacred import Experiment

ex = Experiment('MMSP')


def _loss_names(d):
    ret = {
        'sp': 0,
        'tit': 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = 'mmp'
    seed = 0
    datasets = ['jd']
    loss_names = _loss_names({'sp': 1})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    message = ''
    external_ti_embedding = False
    ti_embedding_size = 2048
    depth = 12

    # Image setting
    max_image_num = 5
    train_transform_keys = ['pixelbert']
    val_transform_keys = ['pixelbert']
    image_size = 800
    max_image_len = -1
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # Text Setting
    word_embedding = False
    max_text_len = 40
    tokenizer = 'bert-base-uncased'
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0
    
    # Attr Setting
    with_attrs = 1  # TODO: 去掉这个参数，以免造成误解
    attr_col_file = ''
    
    # FM settings
    fm_embed_dim = 4
    fm_hidden_dims = [324, 32, 1]
    # len(fm_dropouts) = len(fm_hidden_dims)
    fm_dropouts = [0.8, 0.8, 0.8]
    class_token_size = 324
    with_second_order = True
    with_deep = True
    with_mlp = False
    with_ti_in_fm = True
    second_deep = False

    # Transformer Setting
    vit = 'vit_base_patch32_384'
    hidden_size = 768
    num_heads = 12
    num_layers = 1
    mlp_ratio = 4
    drop_rate = 0.1
    pretrained = True

    # Optimizer Setting
    optim_type = 'adamw'
    learning_rate = 1e-4
    weight_decay = 0.01  # l2 regularization
    decay_power = 1
    max_epoch = 100
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # TIT Task Setting
    tit_loss_factor = 1

    # Downstream Setting
    sp_loss = 'l1_loss'
    get_recall_metric = False
    metrics = ['wmape', 'wmape_region_dt', 'wmape_region', 'wmape_all']

    

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    ckpt_path = None

    # Early Stopping Setting
    patience = 60
    hp_metric = 'wmape_all'
    mode = "min"

    # below params varies with the environment
    workspace_dir = '.'
    data_root = ''
    log_dir = 'result'

    per_gpu_batchsize = 2  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ''
    num_workers = 8
    precision = 32

    # test
    test_version = None

    # test model
    new_model = False


# Named configs for 'environment' which define gpus and nodes, and paths
@ex.named_config
def env_mlops():
    workspace_dir = '/home/jovyan/MMSP/'
    num_gpus = 2
    num_nodes = 1
    batch_size = 512


@ex.named_config
def light_transformer():
    vit = 'vit_base_patch32_384_depth_1'


@ex.named_config
def no_transformer():
    vit = 'vit_base_patch32_384_depth_0'


@ex.named_config
def second_deep():
    second_deep = True


@ex.named_config
def new_arrival_wo_tit():
    attr_col_file = 'new_attr_cols.json'
    loss_names = _loss_names({'sp': 1})
    metrics = ['wmape', 'wmape_all', 'acc', 'mae']
    hp_metric = 'wmape'  # 指定early stopping的monitor，及tensorboard的hparams中汇报的metric    

    
@ex.named_config
def new_arrival():
    attr_col_file = 'new_attr_cols.json'
    loss_names = _loss_names({'sp': 1, 'tit': 1})
    patience = 10
    metrics = ['wmape', 'wmape_all', 'acc', 'mae']  # 指定需要汇报的metrics
    hp_metric = 'wmape'  # 指定early stopping的monitor，及tensorboard的hparams中汇报的metric
    tit_loss_factor = 1


@ex.named_config
def word_embedding():
    word_embedding = True
    max_text_len = 30  # t-shirt集最大为25
    vocab_size = 2500  # 从101开始


@ex.named_config
def without_attrs():
    fm_embed_dim = 0


@ex.named_config
def without_ti():
    class_token_size = 0

@ex.named_config
def external_ti_embedding():
    pretrained = False
    external_ti_embedding = True
    ti_embedding_size = 2048
    depth = 12


@ex.named_config
def tshirt():
    data_root = 'data/arrows/jd_t-shirt_49_1'
    exp_name = 't-shirt'


@ex.named_config
def jeans():
    data_root = 'data/arrows/jd_jeans_49_1'
    exp_name = 'jeans'


@ex.named_config
def shirt():
    data_root = 'data/arrows/jd_shirt_49_1'
    exp_name = 'shirt'


@ex.named_config
def pants():
    data_root = 'data/arrows/jd_pants_49_1'
    exp_name = 'pants'


# 设置jd训练的任务
@ex.named_config
def task_sp():
    # exp settings
    exp_name = 'mmp'
    datasets = ['jd']
    data_root = 'data/arrows/'
    batch_size = 512
    per_gpu_batchsize = 128
    max_epoch = 60
    patience = 60
    max_steps = None
    message = 'default' 

    # env settings
    num_gpus = 4
    num_nodes = 1
    
    # model settings
    loss_names = _loss_names({'sp': 1})
    attr_col_file = 'new_attr_cols.json'

    learning_rate = 0.001
    optim_type = 'Adam'
    weight_decay = 0.0  # l2 regularization   
    
    # deepFM settings
    fm_embed_dim = 16
    fm_hidden_dims = [128, 128] # 层数越深最后的值越小 # [256, 128, 32, 16, 1]
    fm_dropouts = [0.3, 0.3] # [0.5, 0.5, 0.5, 0.5, 0.5]  # TODO: 不dropout，即dropout=0试试。len(fm_dropouts) = len(fm_hidden_dims)
    class_token_size = 16
    with_second_order = True
    with_deep = True
    with_mlp = False
    with_ti_in_fm = True
    second_deep = False

    # image_settings
    train_transform_keys = ['pixelbert_nonresize']
    val_transform_keys = ['pixelbert_nonresize']
    max_image_num = 7 # 每个sku所读取的image文件最大数量，现有版本中不适用
    image_size = 384  # image大小，需要根据图片调节
    pretrained = True

    # text settings
    tokenizer = 'bert-base-chinese'
    whole_word_masking = True