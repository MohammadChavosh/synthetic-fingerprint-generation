import subprocess
import os

from fingerprints_dataset import get_fingerprint_images_list

bashCommand = "/home/chavosh/NBIS/bin/mindtct {} /home/chavosh/extracted_minutiaes/real_from_HR/{}"
images = get_fingerprint_images_list('/home/sadegh/Fingerprint_files/sd09/', load_cropped=True, HR_true=True)
for idx, image in enumerate(images):
    if not image.endswith('_cropped_resized_HR.png'):
        print(image)
        continue
    process = subprocess.Popen(bashCommand.format(image, image.split('/')[-1][:-4]).split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if idx % 1000 == 0:
        print("{}/{} processed".format(idx + 1, len(images)))
