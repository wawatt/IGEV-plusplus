import sys
sys.path.append('core_rt')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from rt_igev_stereo import IGEVStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import skimage.io
import cv2


DEVICE = 'cuda'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_image(imfile):
    img = np.array(Image.open(imfile))[:,:,0:3].astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp)
            file_stem = os.path.join(output_directory, imfile1.split('/')[-1])
            disp = disp.cpu().numpy().squeeze()
            if args.save_png:
                disp_16 = np.round(disp * 256).astype(np.uint16)
                # skimage.io.imsave(file_stem, disp_16)
                plt.imsave(file_stem, disp, cmap='jet')

            if args.save_numpy:
                np.save(file_stem.replace('.png', '.npy'), disp)
                
        dummy_input_left = torch.randn(1, 3, 384, 640).to(DEVICE)  # 左图像
        dummy_input_right = torch.randn(1, 3, 384, 640).to(DEVICE)  # 右图像

        # 导出为 ONNX 格式，并设置输入和输出的名称
        torch.onnx.export(model, 
            (dummy_input_left, dummy_input_right), 
            "IGEVStereo_rt.onnx", 
            verbose=True,
            opset_version=13,
            do_constant_folding=True,
            export_params=True,
            input_names=['left','right'], 
            output_names=['pred_disp'],
            dynamic_axes = None
        )
        import onnx
        import onnxsim
    try:
        model_onnx = onnx.load("IGEVStereo_rt.onnx")  # load onnx model
        # model_opt, check = onnxsim.simplify(model_onnx, include_subgraph=True, skip_shape_inference=True)
        model_opt, check = onnxsim.simplify(model_onnx)
    
        assert check, 'assert check failed'
        onnx.save(model_opt, "IGEVStereo_rt_simplified.onnx")
        print("ok")
    except Exception as e:
        print(f'----------------simplifier failure: {e}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/igev_rt/sceneflow.pth')
    parser.add_argument('--save_png', action='store_true', default=True, help='save output as gray images')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="demo-imgs/sceneflow/*0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="demo-imgs/sceneflow/*1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="output/")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=8, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=96, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp range")
    
    args = parser.parse_args()

    demo(args)
