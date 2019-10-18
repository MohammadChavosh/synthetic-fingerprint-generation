# Synthetic Fingerprint Generation

In this project, we create synthetic fingerprints!

The code consist of two parts, first we create some low-resolution fingerprints using a GAN and then we train another network to make them high quality!

In order to test the results, there is a `extract_minutiae.py` script that can be used to extract minutiae of real images and fake ones. Then, you can train a classifier to assess quality of my fake images. If the classifier can distinguish fake images from real ones, the quality is not good. Code for a simple classifier is in the `Minutiae classifier.ipynb`. You need to have [NBIS tools](https://www.nist.gov/itl/iad/image-group/products-and-services/image-group-open-source-server-nigos#Releases) to extract minutiaes.
