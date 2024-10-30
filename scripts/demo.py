import os
import sys
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
sys.path.append('.')

from GeomConsistentFR import test_relight_single_image_lighting_transfer
from training.config import get_config
from training.inference import Inference
from training.utils import create_logger, print_args

def main(config, args, reference_path, reference_filename):
    logger = create_logger(args.save_folder, args.name, 'info', console=True)
    print_args(args, logger)
    logger.info(config)

    inference = Inference(config, args, args.load_path)

    imga_name = args.source
    imgb_name = reference_path
    imgA = Image.open(imga_name).convert('RGB')
    imgB = Image.open(imgb_name).resize((361,361)).convert('RGB')
    imgC = Image.open(os.path.join(args.reference_dir, reference_filename+'.png')).convert('RGB')

    result = inference.transfer(imgA, imgB, postprocess=True)

    imgA = np.array(imgA.resize((361,361))); imgB = np.array(imgB)
    h, w, _ = imgA.shape
    result = result.resize((h, w))
    result = np.array(result)
    Image.fromarray(result.astype(np.uint8)).save(os.path.join(args.save_folder, f"{reference_filename}.png"))
    vis_image = np.hstack((imgA, imgC, result))
    save_path = os.path.join(args.save_folder, f"{reference_filename}_all.png")
    Image.fromarray(vis_image.astype(np.uint8)).save(save_path)
    print('Finished!')
    return Image.fromarray(result.astype(np.uint8))


def main_start(source,reference,lighting_transfer=True):
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--name", type=str, default='demo')
    parser.add_argument("--save_path", type=str, default='result', help="path to save model")
    parser.add_argument("--load_path", type=str, help="folder to load model",
                        default='ckpts/G.pth')
    parser.add_argument("--source-dir", type=str, default="assets/images/non-makeup")
    parser.add_argument("--reference-dir", type=str, default="assets/images/makeup")
    parser.add_argument("--gpu", default='0', type=str, help="GPU id to use.")
    parser.add_argument("--source", type=str, default=f"{source}")
    parser.add_argument("--reference", type=str, default=f"{reference}")

    args = parser.parse_args()
    args.gpu = 'cuda:' + args.gpu
    args.device = torch.device(args.gpu)

    args.save_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    config = get_config()

    reference_filename = reference.split('/')[-1]
    if lighting_transfer:
        test_relight_single_image_lighting_transfer.main_lighting_trasfer(reference,source,reference_filename)
        reference_filename = reference_filename.replace('.png','')
        reference_path = f'GeomConsistentFR/lighting_transfer_result/{reference_filename}_rendered_image.png'
    else:
        reference_path = reference
        reference_filename = reference.split('/')[-1].replace('.png','')

    result = main(config, args, reference_path, reference_filename)
    return result


if __name__ == "__main__":
    so = "xfsy_0444.png"
    for re in os.listdir('assets/images/makeup'):
        main_start(f'assets/images/non-makeup/{so}', f'assets/images/makeup/{re}', lighting_transfer=True)
        print(f'{re} finished!')

