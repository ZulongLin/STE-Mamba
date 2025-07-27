import os
import csv
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from models.mlp import MLP
from models.STE_Mamba import STE_Mamba_s,STE_Mamba_b,STE_Mamba_l


class MILRegressionModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        ModelClass = globals()[args.model_name]
        self.model = ModelClass(args=args)
        self.mse = nn.MSELoss()
        self.mean_absolute_error = MeanAbsoluteError()
        self.mean_squared_error = MeanSquaredError()
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.validation_step_val_loss = []
        self.validation_step_val_pre = []
        self.validation_step_val_label = []
        self.epoch_start_time = None
        self.start_gpu_memory_usage = None
        self.mlp1 = MLP(input_features=1024)
        self.mlp2 = MLP(input_features=1024)

        self.parameter_csv_path = ""
        self.time_memory_csv_path = ""

    def count_trainable_parameters_mb(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable_params * 4 / (1024 ** 2)

    def save_model_info(self):
        model_name = self.args.model_name
        parameter_count_mb = self.count_trainable_parameters_mb()
        in_len = self.args.in_len
        train_data = self.args.train_data

        file_exists = os.path.exists(self.parameter_csv_path)
        with open(self.parameter_csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Model Name", "Trainable Parameters (MB)", "Input Length (in_len)", "Training Data"])
            writer.writerow([model_name, parameter_count_mb, in_len, train_data])

    def save_time_memory_info(self, epoch_time_ms, memory_usage_mb, max_memory_usage_gb):
        model_name = self.args.model_name
        in_len = self.args.in_len
        train_data = self.args.train_data

        file_exists = os.path.exists(self.time_memory_csv_path)
        with open(self.time_memory_csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Model Name", "Epoch Time (ms)", "GPU Memory Usage (MB)",
                                 "Max GPU Memory Usage (GB)", "Input Length (in_len)", "Training Data"])
            writer.writerow([model_name, epoch_time_ms, memory_usage_mb, max_memory_usage_gb, in_len, train_data])

    def forward(self, x):
        x_time, x_channel = self.model(x.float())
        if self.args.use_channel_mixer:
            y_hat_channel = self.mlp2(x_channel)
        if self.args.use_time_mixer:
            y_hat_time = self.mlp1(x_time)
        if self.args.use_channel_mixer and self.args.use_time_mixer:
            y_hat = (y_hat_time + y_hat_channel) / 2
        else:
            if self.args.use_time_mixer:
                y_hat = y_hat_time
            if self.args.use_channel_mixer:
                y_hat = y_hat_channel

        return y_hat

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()
        self.start_gpu_memory_usage = torch.cuda.memory_allocated() / (1024 ** 2)
        torch.cuda.reset_max_memory_allocated()

    def on_train_epoch_end(self):
        epoch_end_time = time.time()
        end_gpu_memory_usage = torch.cuda.memory_allocated() / (1024 ** 2)
        max_memory_usage_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        epoch_time_ms = (epoch_end_time - self.epoch_start_time)
        memory_usage_mb = end_gpu_memory_usage - self.start_gpu_memory_usage

        self.save_time_memory_info(epoch_time_ms, memory_usage_mb, max_memory_usage_gb)

    def training_step(self, batch, batch_idx):
        if batch_idx == 0 and self.current_epoch == 0:
            self.save_model_info()

        timeindex, x, y = batch
        y_hat= self(x.float())
        loss = F.mse_loss(y_hat, y.view(-1, 1))
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss_epoch",
            },
        }

    def validation_step(self, batch, batch_idx):
        self.evaluation_step(batch)

    def test_step(self, batch, batch_idx):
        self.evaluation_step(batch)

    def on_validation_epoch_end(self):
        self.on_evaluation_epoch_end(stage="val")

    def on_test_epoch_end(self):
        self.on_evaluation_epoch_end(stage="test")

    def evaluation_step(self, batch):
        timeindex, x, y = batch
        B, T, C = x.shape
        y_hat= self(x.float())
        loss = F.mse_loss(y_hat, y.view(-1, 1))
        [self.validation_step_val_pre.append(y_hat[i].item()) for i in range(y_hat.shape[0])]
        [self.validation_step_val_label.append(y[i].item()) for i in range(y.shape[0])]
        self.validation_step_val_loss.append(loss)

    def on_evaluation_epoch_end(self, stage="val"):
        avg_val_loss = torch.stack([x for x in self.validation_step_val_loss]).mean()
        avg_val_mae = self.mean_absolute_error(
            torch.tensor(self.validation_step_val_pre), torch.tensor(self.validation_step_val_label)
        )
        avg_val_rmse = torch.sqrt(
            self.mean_squared_error(
                torch.tensor(self.validation_step_val_pre), torch.tensor(self.validation_step_val_label)
            )
        )
        self.log_dict(
            {
                f"{stage}_loss_epoch": avg_val_loss,
                f"avg_{stage}_mae": avg_val_mae,
                f"avg_{stage}_rmse": avg_val_rmse,
            },
            on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True,
        )
        pd.DataFrame([self.validation_step_val_pre, self.validation_step_val_label]).T.to_excel(
            f"{self.args.test_data[0]}_pre.xlsx", index=False, engine="openpyxl"
        )
        self.validation_step_val_pre.clear()
        self.validation_step_val_label.clear()
        self.validation_step_val_loss.clear()