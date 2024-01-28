# All-In-Focus Image Restoration from a Focal Stack Using a Gated U-Net with Bilinear Upsampling

Note that this file is written in Markdown notation to improve the layout; consider renaming it to .md.

## Using/Testing the Model (```test.py```)

The file "test.py" contains the code for using (testing) our model.

In order to use the model please set the following variables inside test.py:

```python
path_0_1 = "path to a focal plane image 0.1 m above ground"
path_0_4 = "path to a focal plane image 0.4 m above ground"
path_1_5 = "path to a focal plane image 1.5 m above ground"

path_out = "path to where the output image should be written to (including the name of the output image)"
```

The model was trained on an image size of 512 x 512. The order of the images does not actually matter, since we randomized it during training.

The dependencies required to run this model are specified in ```requirements.txt```.  
Please make sure to use python 3.9. The python version cannot be specified in ```requirements.txt```.

Please download the (zipped) model weights from here:
https://drive.google.com/file/d/1vxUIh9G_EfubtyP4h7nOnT8x5TGjxjJ8/view
Please put the model weights in the same folder as ```test.py``` (and name the file "weights" - no extension - if necessary).

After setting the above variables, installing the required dependencies, and downloading the model weights, simply run ```test.py```.
The model weights are automatically loaded from the ```weights``` file.
The model architecture (Gated U-Net with Bilinear Upsampling) is defined in the ```aosnet.py``` file.

## Test Data
Please see the "results" folder for selected examples of integral images as well as the corresponding target and output images (these are also contained in our report). The output of our model for the real focal stack is also provided in the "results" folder. Our test set, i.e., test samples, including the integral images (i.e., input for our model) and the corresponding target and output images can be retrieved from here:
TODO

Our test data, excluding the output images, can also be obtained from here:
https://drive.google.com/file/d/1JCaE4zzmF5Dw4JaJaBaY46EpyXfrTGmy/view
The uncompressed .zip file contains roughly 1.5GB of images, corresponding to 10% of the entire dataset (3200 samples).

We also provide our validation data, which can be obtained from here:
https://drive.google.com/file/d/1jSUXO4N4l6UOH7Uyl0p1_fPa4SE3jSNA/view
The uncompressed .zip file contains roughly 1.5GB of images, corresponding to another 10% of the entire dataset (3200 samples).

## Training/Retraining the Model (```train.py```)

The file "train.py" contains the code for training our model.

The file itself is executable, i.e., it contains an "if __name__ == '__main__':" block.

THIS MUST BE EDITED: PATH_TO_X_DIRECTORY (image samples), PATH_TO_Y_DIRECTORY (targets), and SAVE_PATH.

The dataloader can handle two types of data:
 1. Option 1: Sample and target images (".png") from image samples ("PATH_TO_X_DIRECTORY") and targets directory ("PATH_TO_Y_DIRECTORY").
 2. Option 2: Samples and targets as arrays from one single ".npy" file each (i.e., all samples are contained in one file and all targets are also contained in one - different - file).

Option 1:
We assume that the samples (X) and targets (y) are in different directories.
Samples (X) and targets (y) can be of two formats:
 1. ".png" images (naming convention: idx_[10, 40, 150].png, e.g., 007_10.png for the 7th image with its focal plane at 10 cm).
 2. ".npy" files containing one sample each (e.g., three input images in one array, channels first).

Option 2:
Specify the path to the ".npy" files as indicated in the code (```np.load(...)```).

Other important remarks:
 1. Samples are divided into train (0.8), validation (0.1), and test (0.1) data (function ```divide()```).
 2. ```AOSDataset```: Loads samples on-demand and applies preprocessing steps (histogram equalization, normalization, random augmentations - see our report for details).
 3. ```DataLoader```: PyTorch class which can be iterated over and handles ```batch_size``` and ```shuffle``` - see function ```load_data()```.
 4. The model architecture (Gated U-Net with Bilinear Upsampling) is defined in the ```aosnet.py``` file. (Please keep this file in the same folder as ```train.py```.

Training:
- The number of epochs can be defined in the variable ```num_epochs```.
- We used MSE loss and Adam optimizer with an initial learning rate of 0.001.
- We also used a (```StepLR```) learning rate sheduler with a step_size 0f 5 and a gamma of 0.5 (i.e., halving the learning rate every 5 epochs).
