# All-In-Focus Image Restoration from a Focal Stack Using a Gated U-Net with Bilinear Upsampling

## Using/Testing the Model

In order to use (and test) the model please set the following variables inside test.py:

```python
path_0_1 = "path to the focal plane image of 0.1 m above ground"
path_0_4 = "path to the focal plane image of 0.4 m above ground"
path_1_5 = "path to the focal plane image of 1.5 m above ground"

path_out = "path to where the output image should be written to (including the name of the output image)"
```

After setting the above variables, simply run ```test.py```.
The model weights are automatically loaded from the ```weights``` file.
The model architecture (Gated U-Net with Bilinear Upsampling) is defined in the ```aosnet.py``` file.

## Test Data

Our test data can be obtained from here: https://drive.google.com/file/d/1jSUXO4N4l6UOH7Uyl0p1_fPa4SE3jSNA/view?usp=drive_link
The uncompressed .zip file contains roughly 160 GB of images, corresponding to 10% of the entire dataset.

## Retraining the Model
