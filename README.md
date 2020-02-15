# Synthetic Fingerprint Generation

In this project, we create synthetic fingerprints!

The code consist of two parts, first we create some low-resolution fingerprints using a GAN and then we train another network to make them high quality!

## Data prepration

First, we need to do some pre-processing over the dataset. The images in the `Sd09` fingerprints dataset are not aligned in center and have some noisy artifacts surronding the fingerprint. We need to crop the fingerprint part and clean our dataset.

After putting the dataset in the correct path and installing the [NBIS tools](https://www.nist.gov/itl/iad/image-group/products-and-services/image-group-open-source-server-nigos#Releases) to have `nfseq` bash command, you can easily modify the `fingerprint-generator/crop_images_nfseg.py` to create the clean dataset.

## Training the fingerprint generator

After finishing the dataset pre-processing, you can easily train the fingerprint generator GAN using the `fingerprint-generator/main.py` code. You just need to pass the correct path to it. The code can get several parameters, which you can change to alter the training process. Here is an example of how you can use the command.

```
python main.py --dataset fingerprints --dataroot path/to/sd09 --imageSize 64 --nc 1 --niter 100 —cuda
```
Look at the `main.py` code to see different parameters that you can pass to the script.

## Training the super resulotion network

The output of our generator is low quality fingerprints. We are going to use a super-resolution network to make them better. In order to do so, we need to train a network that make high resolution images from low resolution ones. For that, we need to create a low to high resolution dataset, which can be done by `fingerprint-generator/create_HR_LR_images.py`. You just need to set the path to `Sd09` dataset in it correctly.

Then, you can use `low-to-high-resolution/codes/train.py` to train super-resolution network. This code needs some option file to use for trianing. You just need to set the correct paths in `low-to-high-resolution/codes/options/train/train_sr.json` file. Then, you can train the network via below command:

```
python train.py -opt options/train/train_sr.json
```

After training, in order to run the trained model over our generated low-quality fingerprints to make them high-resolution, you just need to use the `low-to-high-resolution/codes/test.py`. Similar to train, it needs some options which can be found in `low-to-high-resolution/codes/options/test/test_sr.json`. Do not forget to set path correctly there. Here is the test command:

```
python test.py -opt options/test/test_sr.json
```

## Testing the outputs

In order to test the results, there is a `extract_minutiae.py` script that can be used to extract minutiae of real images and fake ones. You just need to set pathes correctly in that. Then, you can train a classifier to assess quality of my fake images. If the classifier can distinguish fake images from real ones, the quality is not good. Code for a simple classifier is in the `Minutiae classifier.ipynb`. You need to have NBIS tools installed to extract minutiaes.
