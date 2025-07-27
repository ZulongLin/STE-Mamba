import sys

import lightning as pl
from lightning import seed_everything
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from lightning.pytorch.loggers import CSVLogger
from dataset import MILRegressionDataModule
import os

from lightning_process import MILRegressionModel

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse

if __name__ == '__main__':
    results = []
    train_dataset = 'AVEC2014'
    test_dataset = 'AVEC2014'

    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--data_dir', default='', type=str,
                        help='data_dir')

    parser.add_argument('--label_file', default='', type=str,
                        help='data_dir')
    parser.add_argument('--train_data', default=[f'{train_dataset}-train'], nargs='+', help='traindata')
    parser.add_argument('--val_data', default=[f'{test_dataset}-dev'], nargs='+', help='valdata')
    parser.add_argument('--test_data', default=[f'{test_dataset}-test'], nargs='+', help='testdata')
    parser.add_argument('--model_name', default='STE_Mamba_s', type=str,
                        help='modals model')
    parser.add_argument('--regression_model', default='MLP', type=str,
                        help='regression_model')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    parser.add_argument('--num_workers', default=0, type=int, help='num_workers')
    parser.add_argument('--max_epochs', default=50, type=int, help='max_epochs')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning_rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay')
    parser.add_argument('--frame_interval', default=1, type=int, help='frame_interval')
    parser.add_argument('--join', default=1, type=int, help='join')
    parser.add_argument('--dropout', type=float, default=0., help='dropout')
    parser.add_argument('--seed', type=float, default=1, help='dropout')
    parser.add_argument('--devices', type=int, default=[0], nargs='+', help='dropout')
    parser.add_argument('--deep', default=True, type=bool, help='deep', )
    parser.add_argument('--deep_type',
                        default='iresnet50_base_1',
                        type=str, help='deep_type')

    parser.add_argument('--audio', action="store_false", help='audio', )
    parser.add_argument('--audio_type', default='audio_pann_64_3', type=str, help='audio_type')

    parser.add_argument('--rppg',default=True, type=bool, help='rppg', )
    parser.add_argument('--rppg_type', default='HRV_3000_normal' if train_dataset == 'AVEC2013' else 'HRV_1000_normal',
                        type=str, help='rppg_type')
    parser.add_argument('--emotion', default=True, type=bool, help='emotion', )
    parser.add_argument('--emotion_type', default='Face_valence_arousal', type=str, help='emotion_type')

    parser.add_argument('--au', default=False, type=bool, help='openface')
    parser.add_argument('--openface_type', default='openface_au', type=str, help='openface_type')

    parser.add_argument('--use_time_mixer', default=True, type=bool, help='if use time mixer')
    parser.add_argument('--use_channel_mixer', default=True, type=bool, help='if channel mixer')
    parser.add_argument('--before_channel_mix_linear_hidden', type=int, default=64, help='patch length')

    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')


    parser.add_argument('--in_len', type=int, default=1000, help='Total length of the input sequence')
    parser.add_argument('--d_ff', type=int, default=256, help='Dimension of the hidden layer')
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of the embedding mapping')
    parser.add_argument('--e_layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--model_dropout', type=int, default=0.1, help='Number of encoder layers')

    parser.add_argument('--input_dims', default=0, type=int, help='attention_dims')
    parser.add_argument('--num_modals', default=0, type=int, help='total number of modals')
    parser.add_argument('--video_dims', default=0, type=int, help='deep_dims')
    parser.add_argument('--audio_dims', default=0, type=int, help='audio_dims')
    parser.add_argument('--rppg_dims', default=0, type=int, help='rppg_dims')

    args = parser.parse_args()
    seed_everything(args.seed)

    print(args)
    data_module = MILRegressionDataModule(args)
    args.in_len = int(args.in_len / args.frame_interval)

    model = MILRegressionModel(args=args)
    early_stopping_callback = EarlyStopping(monitor='val_loss_epoch', mode='min', patience=15)

    class MyTQDMProgressBar(TQDMProgressBar):
        def __init__(self):
            super(MyTQDMProgressBar, self).__init__()

        def init_validation_tqdm(self):
            bar = Tqdm(
                desc=self.validation_description,
                position=0,
                disable=self.is_disabled,
                leave=True,
                dynamic_ncols=True,
                file=sys.stdout,
            )
            return bar

    checkpoint_callback = ModelCheckpoint(
        monitor='avg_val_mae',
        mode='min',
        dirpath=f'best_model/',
        filename='{epoch}-{avg_val_mae:.4f}-{avg_val_rmse:.4f}'
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy='ddp_find_unused_parameters_true',
        logger=CSVLogger(
            save_dir=f'logs/'),
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, MyTQDMProgressBar()],
        log_every_n_steps=15,
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)