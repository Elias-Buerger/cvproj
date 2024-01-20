import cv2
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
import torch
from aosnet import AOSNet

def main():
    #order of fp doesn't actually matter since model was trained on randomized order
    path_0_1 = 'results/real/FP_0.10.png'
    path_0_4 = 'results/real/FP_0.40.png'
    path_1_5 = 'results/real/FP_1.50.png'
    
    path_out = 'results/real/output.png'

    #load focal stack and augmentations into batch with 1 sample
    batch_data = np.empty((1, 6, 512, 512), dtype=np.ubyte)
    for idx, path in enumerate([path_0_1, path_0_4, path_1_5]):
        img = cv2.imread(path) #load image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (512, 512)) #scale image
        batch_data[0, idx * 2] = img
        
        channel = Image.fromarray(img)
        batch_data[0, idx * 2 + 1] = np.array(TF.equalize(channel)) #add equalization augmentation

    batch_data = torch.Tensor(batch_data) / 255.0 #scale to 0-1 range
    batch_data = torch.clamp(batch_data, 0, 1) #clamp as security measure

    #load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AOSNet(6, 1).to(device)
    model.load_state_dict(torch.load('weights'))
    model.eval()

    #run model
    with torch.no_grad():
        output = model(batch_data.to(device))[0, 0] #disregard batch and channel dimension

    #save output
    output = Image.fromarray(output.detach().cpu().numpy() * 255)
    output = output.convert('RGB')
    output.save(path_out)
    

if __name__ == "__main__":
    main()