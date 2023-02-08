"""
This code is taken from https://github.com/YanchaoYang/FDA/blob/master/utils/__init__.py
(Pytorch implementation of our FDA: Fourier Domain Adaptation for Semantic Segmentation
by Yang paper published in CVPR 2020.)
"""
from PIL import Image
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torchvision.transforms as transforms

import os

## This is the adapted implementation of LucasFidon from Github Issues https://github.com/YanchaoYang/FDA/issues/40
def extract_ampl_phase(fft_im):
    # fft_im: size should be b x 3 x h x w
    fft_amp = torch.abs(fft_im)
    fft_pha = torch.angle(fft_im)
    return fft_amp, fft_pha

## This is the adapted implementation of LucasFidon from Github Issues https://github.com/YanchaoYang/FDA/issues/40
def low_freq_mutate(amp_src, amp_trg, L=0.1):
    _, _, h, w = amp_src.size()
    # multiply w by 2 because we have only half the space as rFFT is used
    w *= 2
    # multiply by 0.5 to have the maximum b for L=1 like in the paper
    b = (np.floor(0.5 * np.amin((h, w)) * L)).astype(int)     # get b
    if b > 0:
        # When rFFT is used only half of the space needs to be updated
        # because of the symmetry along the last dimension
        amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]      # top left
        amp_src[:, :, h-b+1:h, 0:b] = amp_trg[:, :, h-b+1:h, 0:b]    # bottom left
    return amp_src


## This is the adapted implementation of LucasFidon from Github Issues https://github.com/YanchaoYang/FDA/issues/40
def FDA_source_to_target(src_img, trg_img, L=0.1):
    # get fft of both source and target
    fft_src = torch.fft.rfft2(src_img.clone(), dim=(-2, -1))
    fft_trg = torch.fft.rfft2(trg_img.clone(), dim=(-2, -1))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase(fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)

    # recompose fft of source
    real = torch.cos(pha_src.clone()) * amp_src_.clone()
    imag = torch.sin(pha_src.clone()) * amp_src_.clone()
    fft_src_ = torch.complex(real=real, imag=imag)

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.fft.irfft2(fft_src_, dim=(-2, -1), s=[imgH, imgW])

    return src_in_trg


def print_FFT_example(image_path_source, image_path_target, output_path):
    """
    Simple function to print out an FDA transformed image (source image with same amplitude as target img)
    """

    train_transforms = A.Compose([A.Resize(256, 256), A.ToFloat(),ToTensorV2()])

    IMG_MEAN = torch.reshape(torch.from_numpy(np.array((0.3052, 0.2198, 0.1685), dtype=np.float32)), (1,3,1,1))

    image_source = Image.open(image_path_source).convert('RGB')
    image_source = np.array(image_source)
    image_target = Image.open(image_path_target).convert('RGB')
    image_target = np.array(image_target)

    image_source = train_transforms(image=image_source)["image"]
    image_target = train_transforms(image=image_target)["image"]

    image_source = torch.unsqueeze(image_source, dim=0)
    image_target = torch.unsqueeze(image_target, dim=0)

    src_in_trg = FDA_source_to_target(image_source, image_target, L=0.01)

    B, _, H, W = image_source.shape
    mean_img = IMG_MEAN.repeat(B,1,H,W)

    src_in_trg = src_in_trg.clone() - mean_img

    save_image(os.path.join(output_path, image_source), "image_source.png")
    save_image(os.path.join(output_path, image_target), "image_target.png")
    save_image(os.path.join(output_path, src_in_trg), "src_in_trg.png")