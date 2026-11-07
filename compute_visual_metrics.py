from tqdm import tqdm
import numpy as np
import os
from PIL import Image
import lpips
import matplotlib.pyplot as plt
import numpy as np
import argparse
from natsort import natsorted

parser = argparse.ArgumentParser()
parser.add_argument('--eval_dir', default='test_data')
parser.add_argument('--suffix', default='image_2_usd')
args = parser.parse_args()

loss_fn_alex = lpips.LPIPS(net='alex').cuda()
eval_folder = args.eval_dir
mses = []
lpips_dists = []
tot_mse, tot_lpips = 0, 0
total = 0
max_objects = natsorted(os.listdir(eval_folder))
for obj in tqdm(max_objects, desc='Object Count'):
    mse = 0
    lpips_dist = 0
    num = 0
    samples = natsorted(os.listdir(os.path.join(eval_folder, obj)))
    for sample in tqdm(samples, desc='Samples'):
        pred_image = os.path.join(eval_folder, obj, sample, f'pred_{args.suffix}.png')
        label_image = os.path.join(eval_folder, obj, sample, f'image.png')
        if os.path.exists(pred_image):
            img_pred = np.asarray(Image.open(pred_image), dtype=np.float64) / 255
            img_label = np.asarray(Image.open(label_image), dtype=np.float64) / 255
            mse += np.mean((img_pred - img_label)**2, dtype=np.float64)
            label_image_cuda = lpips.im2tensor(lpips.load_image(label_image)).cuda()
            pred_image_cuda = lpips.im2tensor(lpips.load_image(pred_image)).cuda()
            lpips_dist += loss_fn_alex.forward(label_image_cuda, pred_image_cuda).item()
            num += 1
    mses.append(mse/max(1, num))
    lpips_dists.append(lpips_dist/max(1, num))
    tot_mse += mse
    tot_lpips += lpips_dist
    total += num
print(f'MSE: {(tot_mse / total).item()}')
print(f'LPIPS: {tot_lpips / total}')       
