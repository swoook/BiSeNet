
import sys
sys.path.insert(0, '.')
import argparse
import time

import torch
import numpy as np
import cv2

import lib.transform_cv2 as T
from lib.models import model_factory
from lib.misc import letterbox_resize
from configs import set_cfg_from_file

from tqdm import trange, tqdm


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    np.random.seed(123)

    # args
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', type=str, default='configs/bisenetv2.py',)
    parse.add_argument('--weight-path', type=str, default='./res/model_final.pth',)
    parse.add_argument('--img-path', dest='img_path', type=str, default='./example.png',)
    args = parse.parse_args()
    cfg = set_cfg_from_file(args.config)


    palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

    # define model
    net = model_factory[cfg.model_type](n_classes=19, is_train=False)
    net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
    net.eval()
    net.cuda()

    # prepare data
    to_tensor = T.ToTensor(
        mean=(0.3257, 0.3690, 0.3223), # city, rgb
        std=(0.2112, 0.2148, 0.2115),
    )
    im = cv2.imread(args.img_path)[:, :, ::-1]
    im = letterbox_resize(src=im, dst_h=512, dst_w=512)
    cv2.imwrite('./input.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

    # inference
    out = net(im).squeeze().detach().cpu().numpy()
    pred = palette[out]
    cv2.imwrite('./res.jpg', pred)

    print("start warm up")
    num_repet = 10
    with torch.no_grad():
        dummy_inputs = torch.rand(num_repet, 3, 512, 512).cuda()
    for idx in range(100 if num_repet >= 100 else num_repet): pred = net(dummy_inputs[idx:idx+1])
    print("warm up done")

    torch.cuda.synchronize()
    time_s = time.perf_counter()
    for idx in range(num_repet): net(dummy_inputs[idx:idx+1])[0].argmax(dim=1).squeeze()#.detach().cpu().numpy()
    torch.cuda.synchronize()
    time_end = time.perf_counter()
    inference_time = (time_end - time_s) / num_repet
    print('FPS: {}'.format((1/inference_time)))
