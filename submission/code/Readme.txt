# All-In-Focus Image Restoration from a Focal Stack Using a Gated U-Net with Bilinear Upsampling

Note that this file is written in Markdown notation to improve the layout; consider renaming it to .md.

## Using/Testing the Model

In order to use (and test) the model please set the following variables inside test.py:

```python
path_0_1 = "path to a focal plane image 0.1 m above ground"
path_0_4 = "path to a focal plane image 0.4 m above ground"
path_1_5 = "path to a focal plane image 1.5 m above ground"

path_out = "path to where the output image should be written to (including the name of the output image)"
```

The model was trained on an image size of 512 x 512. The order of the images does not actually matter, since we randomized it during training.

The dpenedencies required to run this model are specified in ```requirements.txt```.  
Please make sure to use python 3.9. Sadly this can not be set inside ```requirements.txt```.

After setting the above variables and installing the dependencies, simply run ```test.py```.
The model weights are automatically loaded from the ```weights``` file.
The model architecture (Gated U-Net with Bilinear Upsampling) is defined in the ```aosnet.py``` file.

## Test Data

Our test data can be obtained from here: https://drive.google.com/file/d/1JCaE4zzmF5Dw4JaJaBaY46EpyXfrTGmy/view?usp=drive_link.  
The uncompressed .zip file contains roughly 1.5GB of images, corresponding to 10% of the entire dataset.

Our validation data can be obtained from here: https://drive.google.com/file/d/1jSUXO4N4l6UOH7Uyl0p1_fPa4SE3jSNA/view?usp=drive_link.  
The uncompressed .zip file contains roughly 1.5GB of images, corresponding to another 10% of the entire dataset.

## Retraining the Model

TODO
