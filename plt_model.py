import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

class Wrapper(pl.LightningModule):
    def __init__(self, model, learning_rate=2e-5, epochs=5):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tosave = False
        self.saved = []

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch["input_ids"], 
                            attention_mask=batch["attention_mask"], 
                            labels=batch["labels"])
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    # 移除sync_dist=True参数
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log('val_loss', loss, prog_bar=True)  # 移除sync_dist=True
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log('test_loss', loss, prog_bar=True)  # 移除sync_dist=True
        return loss

    # 添加缺失的_shared_step方法
    def _shared_step(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        return outputs.loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        
        # 计算总训练步数
        num_training_steps = self.trainer.estimated_stepping_batches
        
        # 创建学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }