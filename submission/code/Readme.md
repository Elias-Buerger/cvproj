# All-In-Focus Image Restoration from a Focal Stack with a Gated U-Net with Bilinear Upsampling

## running the model

In order to test the model please set the following variables inside test.py:

```python
path_0_1 = 'path to the focal plane image of 10cm above the ground'  
path_0_4 = 'path to the focal plane image of 0.4m above the ground'  
path_1_5 = 'path to the focal plane image of 1.5m above the ground'  

path_out = 'path to where the output image should be written to (including the final image name)'
```

Finally simply run ```test.py```.
The model weights are automatically loaded from the ```weights``` file.

## test data set

Our test data (used for validation) can be found here https://drive.google.com/file/d/1jSUXO4N4l6UOH7Uyl0p1_fPa4SE3jSNA/view?usp=drive_link.  
The uncompressed zip contains roughly 160GB of images and contains 10% of the entire dataset.