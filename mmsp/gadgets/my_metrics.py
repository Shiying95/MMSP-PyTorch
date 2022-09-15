import torch
from pytorch_lightning.metrics import Metric
import torch.nn.functional as F


class Acc(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")        
    
    def update(self, preds, labels, mask):
        preds = preds.detach().to(self.total.device)
        labels = labels.detach().to(self.total.device)
        mask = mask.detach().to(self.total.device)

        abs_error = torch.abs(preds - labels)

        flag = torch.where(
            ((labels <= 10.0) & (abs_error <= 2.0)) | (abs_error / labels <=0.2),
            torch.tensor(1.0, device=self.total.device),
            torch.tensor(0.0, device=self.total.device))

        # flag = torch.where(
        #     (abs_error / labels <=0.3),
        #     torch.tensor(1.0, device=self.total.device),
        #     torch.tensor(0.0, device=self.total.device))

        mask_flag = flag.mul(mask)  # 将被mask的部分置0

        self.correct += mask_flag.sum()
        self.total += mask.sum()  # 分母为有效数据点

    def compute(self):
        return self.correct / self.total


class MAE(Metric):
    '''
    单独对sku-day取wmape的均值。理论上数值大于wmape_all。
    '''
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")        
    
    def update(self, preds, labels, mask):
        preds = preds.detach().to(self.abs_error.device)
        labels = labels.detach().to(self.abs_error.device)
        mask = mask.detach().to(self.abs_error.device)

        mask_preds = preds.mul(mask)  # 无数据部分置零
        mask_labels = labels.mul(mask)  # 无数据部分置零
        
        self.abs_error += F.l1_loss(mask_preds, mask_labels, reduction='sum')
        self.total += mask.sum()  # 加权分母

    def compute(self):
        return self.abs_error / self.total


class Wmape(Metric):
    '''
    单独对sku-day取wmape的均值。理论上数值大于wmape_all。
    '''
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")        
    
    def update(self, preds, labels, mask):
        preds = preds.detach().to(self.abs_error.device)
        labels = labels.detach().to(self.abs_error.device)
        mask = mask.detach().to(self.abs_error.device)

        mask_preds = preds.mul(mask)  # 无数据部分置零
        mask_labels = labels.mul(mask)  # 无数据部分置零
        
        self.abs_error += F.l1_loss(mask_preds, mask_labels, reduction='sum')
        self.total += mask_labels.sum()  # 加权分母

    def compute(self):
        return self.abs_error / self.total


class Wmape_all(Metric):
    '''
    将同一sku下不同天数的总销量求wmape。
    '''
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")        
    
    def update(self, preds, labels, mask):
        preds = preds.detach().to(self.abs_error.device)
        labels = labels.detach().to(self.abs_error.device)
        mask = mask.detach().to(self.abs_error.device)

        mask_preds = preds.mul(mask)  # 无数据部分置零
        mask_labels = labels.mul(mask)  # 无数据部分置零

        sum_mask_preds = torch.sum(mask_preds, dim=-1)
        sum_mask_labels = torch.sum(mask_labels, dim=-1)

        self.abs_error += F.l1_loss(sum_mask_preds, sum_mask_labels, reduction='sum')
        self.total += sum_mask_labels.sum()  # 加权分母

    def compute(self):
        return self.abs_error / self.total


class Wmape_region_dt(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")        
    
    def update(self, preds, labels, mask, region, dt):
        preds = preds.detach().to(self.abs_error.device)  # (None, 1, maxlen)
        labels = labels.detach().to(self.abs_error.device)
        mask = mask.detach().to(self.abs_error.device)
        region = region.detach().to(self.abs_error.device)  # (None, 1, maxlen)
        dt = dt.detach().to(self.abs_error.device)  # (None, 1, maxlen)

        mask_preds = preds.mul(mask)  # 无数据部分置零
        mask_labels = labels.mul(mask)  # 无数据部分置零

        unique_dt = dt.unique()
        unique_region = region.unique()

        for i in unique_region:
            if i==torch.tensor(0):  # 略去mask值
                continue
            for j in unique_dt:
                if j==torch.tensor(0):  # 略去mask值
                    continue
                region_dt_mask = torch.where(
                    (region==i) & (dt==j), 
                    torch.tensor(1.0, device=self.abs_error.device), 
                    torch.tensor(0.0, device=self.abs_error.device),
                    )
                # print(f'region-dt {i}-{j} have {region_dt_mask.sum()}/{mask.sum()} valid records.')
                
                # 只取指定region&dt的预测结果
                region_dt_preds = mask_preds.mul(region_dt_mask).sum(axis=-1)  # (None, 1)
                region_dt_labels = mask_labels.mul(region_dt_mask).sum(axis=-1) # (None, 1)
                # print(f'region_dt preds: {region_dt_preds}, region_dt labels: {region_dt_labels}')

                self.abs_error += F.l1_loss(region_dt_preds, region_dt_labels, reduction='sum')
                self.total += region_dt_labels.sum()  # 真实销量和作为分母
                # print(f'abs_error: {self.abs_error}, total: {self.total}')

    def compute(self):
        return self.abs_error / self.total


class Wmape_region(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")        
    
    def update(self, preds, labels, mask, region):
        preds = preds.detach().to(self.abs_error.device)  # (None, 1, maxlen)
        labels = labels.detach().to(self.abs_error.device)
        mask = mask.detach().to(self.abs_error.device)
        region = region.detach().to(self.abs_error.device)  # (None, 1, maxlen)
        
        mask_preds = preds.mul(mask)  # 无数据部分置零
        mask_labels = labels.mul(mask)  # 无数据部分置零

        unique_region = region.unique()
        for i in unique_region:
            if i==torch.tensor(0):
                continue
            region_mask = torch.where(
                region==i, 
                torch.tensor(1.0, device=self.abs_error.device), 
                torch.tensor(0.0, device=self.abs_error.device),
                )
            # print(f'region {i} have {region_mask.sum()}/{mask.sum()} valid records.')
            
            # 只取指定region的预测结果
            region_preds = mask_preds.mul(region_mask).sum(axis=-1)  # (None, 1)
            region_labels = mask_labels.mul(region_mask).sum(axis=-1) # (None, 1)
            # print(f'region preds: {region_preds}, region labels: {region_labels}')

            self.abs_error += F.l1_loss(region_preds, region_labels, reduction='sum')
            self.total += region_labels.sum()  # 真实销量和作为分母
            # print(f'abs_error: {self.abs_error}, total: {self.total}')

    def compute(self):
        return self.abs_error / self.total


class Mask_mse_loss(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("mse_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def update(self, preds, labels, mask):
        preds = preds.detach().to(self.mse_loss.device)
        labels = labels.detach().to(self.mse_loss.device)
        mask = mask.detach().to(self.mse_loss.device)

        mask_preds = preds.mul(mask)  # 无数据部分置零
        mask_labels = labels.mul(mask)  # 无数据部分置零

        self.mse_loss += F.mse_loss(mask_preds, mask_labels, reduction='sum')
        self.total += mask.sum()  # 有效数据点数

        # print(f'update mse_loss: {self.mse_loss}, update total: {self.total}')

    def compute(self):
        return self.mse_loss / self.total   



class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        preds = logits.argmax(dim=-1)
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)  # .detach()无需计算梯度
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total
