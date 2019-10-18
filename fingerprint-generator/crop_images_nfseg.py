import subprocess
import numpy as np
import os
from PIL import Image
from torchvision.transforms.functional import crop as torch_crop
from fingerprints_dataset import get_fingerprint_images_list


def pil2numpy(x):
    return np.array(x).astype(np.float32)


def save_cropped_image(path, param_dict):
    img = Image.open(path)
    left = param_dict['sx'] - param_dict['sw'] / 2
    low = param_dict['sy'] - param_dict['sh'] / 2
    cropped = torch_crop(img, low, left, param_dict['sh'], param_dict['sw'])
    cropped.save(path[:-4] + "_cropped.png", "PNG")


bashCommand = "/home/chavosh/NBIS/bin/nfseg 1 1 1 3 1 {}"
images = get_fingerprint_images_list('/home/sadegh/Fingerprint_files/sd09/', load_cropped=False)
raw_path = '/home/chavosh/WassersteinGAN'
for idx, image in enumerate(images):
    process = subprocess.Popen(bashCommand.format(image).split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    output = output.split()
    out_dict = {}
    for inner_idx, out in enumerate(output[:-1]):
        try:
            out_dict[out] = int(output[inner_idx + 1])
        except Exception as e:
            continue
    os.remove(os.path.join(raw_path, image.split('/')[-1][:-4] + '_01.raw'))
    if out_dict['e'] > 0:
        print('ERROR in {}'.format(image))
        continue
    save_cropped_image(image, out_dict)
    if idx % 1000 == 0:
        print("{}/{} processed".format(idx + 1, len(images)))
