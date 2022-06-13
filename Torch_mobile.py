import argparse
from loguru import logger
import os
import pathlib

import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
import argparse
import os

import json
import torch
import logging
import traceback

from pytorch_lightning import Trainer

from FOTS.model.model import FOTSModel
from FOTS.utils.bbox import Toolbox

import easydict
from FOTS.data_loader.data_module import ICDARDataModule
import cv2
import numpy as np
torch.cuda.empty_cache()
torch.cuda.memory_summary()
logging.basicConfig(level=logging.DEBUG, format='')





def load_model(model_path, config ,resume = True):
    model = FOTSModel(config)

    if config.data_loader.dataset == 'synth800k':
        data_module = SynthTextDataModule(config)
    else:
        data_module = ICDARDataModule(config)

    root_dir = str(pathlib.Path(config.trainer.save_dir).absolute() / config.name)
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
    if os.path.exists(resume_ckpt):
        checkpoint = torch.load(resume_ckpt)
        model.load_state_dict(checkpoint)
    return model


def main(args:argparse.Namespace):
    model_path = args.model
    input_dir = args.input_dir
    output_dir = args.output_dir
    with_image = True if output_dir else False
    with_gpu = True if torch.cuda.is_available() else False

    config = json.load(open(args.config))
    #with_gpu = False


    config = easydict.EasyDict(config)
    # model = FOTSModel.load_from_checkpoint(checkpoint_path=model_path,
    #                                        map_location='cpu', config=config)
    model = load_model(model_path,config,True)
    model = model.to('cuda:0')
    model.eval()
    image_fn = next(input_dir.glob('*.jpg'))
    # try:
    im = cv2.imread(str(image_fn))[:, :, ::-1]
    
        #im_resized, (ratio_h, ratio_w) = Toolbox.resize_image(im)

    h, w, _ = im.shape
    im_resized = cv2.resize(im, dsize=(640, 640))

    ratio_w = w / 640
    ratio_h = h / 640

    # im_resized = datautils.normalize_iamge(im_resized)

    im_resized = torch.from_numpy(im_resized).float()
    # if True:
    im_resized = im_resized.to('cuda:0')

    im_resized = im_resized.unsqueeze(0)
    im_resized = im_resized.permute(0, 3, 1, 2)

    traced_script_module = torch.jit.trace(model, im_resized)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter("model.ptl")

if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='epoch=38-step=935.ckpt', type=pathlib.Path,
                        help='path to model')
    parser.add_argument('-o', '--output_dir', default='F:/project_2/New_folder/output', type=pathlib.Path,
                        help='output dir for drawn images')
    parser.add_argument('-i', '--input_dir', default='F:/project_2/New_folder/test/', type=pathlib.Path, required=False,
                        help='dir for input images')
    parser.add_argument('-c', '--config', default='pretrain.json', type=str,
                        help='config file path (default: None)')
    args = parser.parse_args('')
    
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
    main(args)
    # model.eval()
    # example = torch.rand(1, 3, 640, 640)
    # example = example.cuda()
    # traced_script_module = torch.jit.trace(model, example)
    # print("this script is run over 50% and it is eval once")
    # traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    # traced_script_module_optimized._save_for_lite_interpreter("model.ptl")
