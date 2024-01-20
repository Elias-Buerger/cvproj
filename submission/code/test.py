import cv2
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
import torch
from aosnet import AOSNet

def main():

    ###########################
    ### CUSTOMIZATION START ###

    # Instructions for using the model are provided in the "Readme.txt" file
    # Order of the focal planes does not actually matter, as the model was trained using a randomized order to match the requirements specified in the exercise session

    # INPUT
    # NOTE: The model requires three images (prefereably the specified focal planes) as input!
    path_0_1 = "results/real/FP_0.10.png" # Focal plane image 0.1 m above ground
    path_0_4 = "results/real/FP_0.40.png" # Focal plane image 0.4 m above ground
    path_1_5 = "results/real/FP_1.50.png" # Focal plane image 1.5 m above ground

    # OUTPUT
    path_out = "results/real/output.png"# File path including the file name

    ### CUSTOMIZATION END ####
    ##########################

    # Load the focal stack and apply fixed augmentation to each channel (focal plane) to produce a batch of one sample
    batch_data = np.empty((1, 6, 512, 512), dtype=np.ubyte)
    for idx, path in enumerate([path_0_1, path_0_4, path_1_5]):
        img = cv2.imread(path) # Load focal plane (image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (512, 512)) # Scale image (if not 512 x 512)
        batch_data[0, idx * 2] = img
        
        channel = Image.fromarray(img)
        batch_data[0, idx * 2 + 1] = np.array(TF.equalize(channel)) # Apply histogram equalization as fixed augmentation technique

    batch_data = torch.Tensor(batch_data) / 255.0 # Scale to [0, 1] range
    batch_data = torch.clamp(batch_data, 0, 1) # Clip as security measure

    # Intitialize model and load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AOSNet(6, 1).to(device)
    model.load_state_dict(torch.load("weights"))
    model.eval()

    # Run model
    with torch.no_grad():
        output = model(batch_data.to(device))[0, 0] # Disregard batch and channel dimension

    # Post-process and save output (to specified location)
    output = (np.clip(output.detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8) # Clip to range [0, 1] and revert normalization (* 255)
    output = Image.fromarray(output)
    output = output.convert("RGB")
    output.save(path_out)

if __name__ == "__main__":
    main()
