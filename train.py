import argparse
import json
from loguru import logger
import os
import pathlib

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from easydict import EasyDict
from torch.hub import load_state_dict_from_url
from torch.utils.data import dataloader
from torch.optim.lr_scheduler import  ReduceLROnPlateau
from FOTS.model.model import FOTSModel
from FOTS.model.loss import *
from FOTS.model.metric import *
from FOTS.data_loader.data_module import SynthTextDataModule, ICDARDataModule
from pytorch_lightning.callbacks import Callback

torch.distributed.is_available()
n_fold= 10
class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print('training epoch begin:')
        # root_dir ="F:/FOTS.PyTorch/saved/pretrain"
        # file_save=root_dir + '/checkpoints/'+"epoch="+str(pl_module.current_epoch)+'-' +"step="+ str(pl_module.global_step-1)+".ckpt"
        # config["pretrain"]= file_save
        # # config.write('pretrain.json') 
        # with open('pretrain1.json','w' ) as f:
        #     f.write(json.dumps(config, indent = 4))
    def on_train_end(self, trainer, pl_module):
        # trainer.datamodule.cross_validation()
        root_dir ="F:/FOTS.PyTorch/saved/pretrain"
        file_save=root_dir + '/checkpoints/'+"epoch="+str(pl_module.current_epoch)+'-' +"step="+ str(pl_module.global_step-1)+".ckpt"
        config["pretrain"]= file_save
        # config.write('pretrain.json') 
        # with open('pretrain1.json','w' ) as f:
        #     f.write(json.dumps(config, indent = 4))

def main(config, resume: bool):
    if not config.cuda:
        gpus = 0
        # device = torch.device('cuda:0')
    else:
        gpus = config.gpus
        # device = torch.device('cuda:0')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FOTSModel(config).to(device)
    # model=torch.load('epoch=38-step=935.ckpt', map_location="cuda:0")
    # model.load_from_checkpoint('epoch=38-step=935.ckpt')
    if resume:
        assert pathlib.Path(config.pretrain).exists()
        resume_ckpt = config.pretrain
        logger.info('Resume training from: {}'.format(config.pretrain))
    else:
        if config.pretrain:
            assert pathlib.Path(config.pretrain).exists()
            logger.info('Finetune with: {}'.format(config.pretrain))
            model.load_from_checkpoint(config.pretrain, config=config, map_location='cpu')
            resume_ckpt = None
        else:
            resume_ckpt = None
    # scheduler = ReduceLROnPlateau(mode='min')
    if config.data_loader.dataset == 'synth800k':
        data_module = SynthTextDataModule(config)
    else:
        data_module = ICDARDataModule(config)
    data_module.setup()
    # data_module.cross_validation()

    root_dir = str(pathlib.Path(config.trainer.save_dir).absolute() / config.name)
    checkpoint_callback = ModelCheckpoint(dirpath=root_dir + '/checkpoints', period=1)
    wandb_dir = pathlib.Path(root_dir) / 'wandb'
    if not wandb_dir.exists():
        wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb_logger = WandbLogger(name=config.name,
                            project='FOTS',
                            config=config,
                            save_dir=root_dir)

    trainer = Trainer(
        # logger=wandb_logger,
        callbacks=[MyPrintingCallback(),checkpoint_callback],
        max_epochs=config.trainer.epochs,
        default_root_dir=root_dir,
        gpus=gpus,
        # accelerator='ddp',
        benchmark=True,
        sync_batchnorm=True,
        precision=config.precision,
        log_gpu_memory=config.trainer.log_gpu_memory,
        log_every_n_steps=config.trainer.log_every_n_steps,
        overfit_batches=config.trainer.overfit_batches,
        # weights_summary='full',
        terminate_on_nan=config.trainer.terminate_on_nan,
        # fast_dev_run=config.trainer.fast_dev_run,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        resume_from_checkpoint=resume_ckpt
        )
    # model= model.to(device)
    trainer.fit(model=model, datamodule=data_module)
    # input_data= data_module.train_dataloader()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default='pretrain1.json', type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume',default=True, action='store_true',
                        help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    config = None
    if args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        # assert not os.path.exists(path), "Path {} already exists!".format(path)
    else:
        if args.resume is not None:
            logger.warning('Warning: --config overridden by --resume')
            config = torch.load(args.resume, map_location='cpu')['config']

    assert config is not None
    config = EasyDict(config)
    main(config, args.resume)
