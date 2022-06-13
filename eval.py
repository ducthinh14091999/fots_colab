import argparse
import os

import json
import torch
import logging
import pathlib
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
    for image_fn in input_dir.glob('*.jpg'):
        # try:
        img = cv2.imread(str(image_fn))
        with torch.no_grad():
            print(image_fn)
            ploy, im = Toolbox.predict(image_fn, model, with_image, output_dir, with_gpu=True)
            print(len(ploy))
        with open("F:/project_2/New_folder/combile_word.txt",'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n').split('*')
                x1,y1,x2,y2,x3,y3, x4,y4 = np.int32(line[2:])
                if line[0].split('/')[-1] == str(image_fn).split('\\')[-1]:
                    img = cv2.polylines(img,[np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]],np.int32)],True,color= [255,0,0],thickness= 2)
                else:
                    cv2.imshow('img',img)
                    cv2.waitKey()
                    break
        # except Exception as e:
        #     traceback.print_exc()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='epoch=330-step=2105.ckpt', type=pathlib.Path,
                        help='path to model')
    parser.add_argument('-o', '--output_dir', default='F:/project_2/New_folder/output', type=pathlib.Path,
                        help='output dir for drawn images')
    parser.add_argument('-i', '--input_dir', default='F:/project_2/New_folder/test/', type=pathlib.Path, required=False,
                        help='dir for input images')
    parser.add_argument('-c', '--config', default='pretrain.json', type=str,
                        help='config file path (default: None)')
    args = parser.parse_args('')
    main(args)









