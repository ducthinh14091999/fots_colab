import argparse
import json
from loguru import logger
import os
import pathlib
from sklearn.model_selection import GridSearchCV
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
import torch.optim as optim
cross_valid =True
from skorch import NeuralNetClassifier


def main(config, resume: bool):
    loss_model = FOTSLoss(config)
    params = {
    'lr': [0.001,0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
    'max_epochs': list(range(500,5500, 500)),
    'batch_size': [4,8,16]
    }
    if not config.cuda:
        gpus = 0
        # device = torch.device('cuda:0')
    else:
        gpus = config.gpus
        # device = torch.device('cuda:0')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FOTSModel(config).to(device)
    net=NeuralNetClassifier(model,max_epochs=20,lr=0.1)
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
    if cross_valid and config.data_loader.dataset != 'synth800k':
        data_module.cross_validation()

    root_dir = str(pathlib.Path(config.trainer.save_dir).absolute() / config.name)
    checkpoint_callback = ModelCheckpoint(dirpath=root_dir + '/checkpoints', period=1)
    wandb_dir = pathlib.Path(root_dir) / 'wandb'
    if not wandb_dir.exists():
        wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb_logger = WandbLogger(name=config.name,
                            project='FOTS',
                            config=config,
                            save_dir=root_dir)
    if os.path.exists(resume_ckpt):
        checkpoint = torch.load(resume_ckpt)
        model.load_state_dict(checkpoint['state_dict'])
    
    # trainer = Trainer(
    #     # logger=wandb_logger,
    #     callbacks=[MyPrintingCallback(),checkpoint_callback],
    #     max_epochs=config.trainer.epochs,
    #     default_root_dir=root_dir,
    #     gpus=gpus,
    #     # accelerator='ddp',
    #     benchmark=True,
    #     sync_batchnorm=True,
    #     precision=config.precision,
    #     log_gpu_memory=config.trainer.log_gpu_memory,
    #     log_every_n_steps=config.trainer.log_every_n_steps,
    #     overfit_batches=config.trainer.overfit_batches,
    #     # weights_summary='full',
    #     terminate_on_nan=config.trainer.terminate_on_nan,
    #     # fast_dev_run=config.trainer.fast_dev_run,
    #     check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
    #     resume_from_checkpoint=resume_ckpt
    #     )
    # # model= model.to(device)
    # trainer.fit(model=model, datamodule=data_module)
    # # input_data= data_module.train_dataloader()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(300):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(iter(data_module.train_dataloader())):
            # get the inputs; data is a list of [inputs, labels]
            image_name,images,score_map,geo_map,training_marks,transcrips,bboxes,rois = data['image_names'], data['images'],data['score_maps'], data['geo_maps'],data['training_masks'], data['transcripts'], data['bboxes'], data['rois']
            images = images.to(device="cuda")
            score_map = score_map.to(device="cuda")
            geo_map = geo_map.to(device="cuda")
            bboxes = bboxes.to(device="cuda")
            rois = rois.to(device="cuda")
            training_marks = training_marks.to(device="cuda")
            # zero the parameter gradients
            optimizer.zero_grad()
            label= [i for i in zip(bboxes,rois) ]
            net.fit(images,label)
            # forward + backward + optimize
            outputs = model.forward(images,bboxes,rois)
            gs = GridSearchCV(model, params, refit=False, scoring='r2', verbose=1, cv=10)

            # gs.fit(images,[bboxes,rois])
            print("Best: %f using %s" % (gs.best_score_, gs.best_params_))
            means = gs.cv_results_['mean_test_score']
            stds = gs.cv_results_['std_test_score']
            params = gs.cv_results_['params']

            if config.data_loader.dataset == 'synth800k':
                # transcrips[1]=transcrips[1][outputs["indices"]]
                # transcrips[0]=transcrips[0][outputs["indices"]]
                transcrips = [transcrips[0][outputs["indices"]],transcrips[1][outputs["indices"]]]
            else:
                transcrips[1]=transcrips[1][outputs["indices"]]
                transcrips[0]=transcrips[0][outputs["indices"]]
            loss_dict = loss_model(score_map,outputs['score_maps'],geo_map,outputs['geo_maps'],transcrips,outputs['transcripts'],training_marks)
            loss = loss_dict['reg_loss'] + loss_dict['cls_loss'] + loss_dict['recog_loss']
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
             # print every 2000 mini-batches
            print('[epoch : {epoch}] recog_loss : {recog_loss} loss cls {cls} and reg_loss {reg}'.format(epoch=epoch, recog_loss=loss_dict['recog_loss'] ,cls= loss_dict['cls_loss'],reg= loss_dict['reg_loss']))
            
            if epoch % 10 ==0:
                torch.save(model.state_dict(),"F:/FOTS.PyTorch/epoch=38-step=935.ckpt")


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