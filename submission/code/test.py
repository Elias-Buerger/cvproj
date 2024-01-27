import cv2
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
import torch
import argparse
from aosnet import AOSNet

def main(weights_path_or_model, focal_plane_1_path, focal_plane_2_path, focal_plane_3_path, path_out):
    """
    Run the AOSNet model on a set of focal plane images.
    This function can either take a path to the model weights or a pre-loaded model object.

    Args:
    weights_path_or_model: Either the path to the model weights or an instance of the AOSNet model.
    focal_plane_1_path: Path to the focal plane image at 0.1m above ground.
    focal_plane_2_path: Path to the focal plane image at 0.4m above ground.
    focal_plane_3_path: Path to the focal plane image at 1.5m above ground.
    path_out: Path for saving the output image.

    The order of the focal planes does not actually matter, as the model was trained using a randomized order.
    """

    # Initialize model and load weights if necessary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(weights_path_or_model, str):
        model = AOSNet(6, 1).to(device)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(weights_path_or_model))
        else:
            model.load_state_dict(torch.load(weights_path_or_model, map_location=torch.device("cpu")))
    elif isinstance(weights_path_or_model, AOSNet):
        model = weights_path_or_model.to(device)
    else:
        raise ValueError("weights_path_or_model must be either a path to the model weights or an AOSNet model instance")

    model.eval()

    # Load the focal stack and apply fixed augmentation to each channel (focal plane) to produce a batch of one sample
    batch_data = np.empty((1, 6, 512, 512), dtype=np.ubyte)
    for idx, path in enumerate([focal_plane_1_path, focal_plane_2_path, focal_plane_3_path]):
        img = cv2.imread(path)  # Load focal plane (image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (512, 512))  # Scale image (if not 512 x 512)
        batch_data[0, idx * 2] = img

        channel = Image.fromarray(img)
        batch_data[0, idx * 2 + 1] = np.array(
            TF.equalize(channel))  # Apply histogram equalization as fixed augmentation technique

    batch_data = torch.Tensor(batch_data) / 255.0  # Scale to [0, 1] range
    batch_data = torch.clamp(batch_data, 0, 1)  # Clip as a security measure

    # Run model
    with torch.no_grad():
        output = model(batch_data.to(device))[0, 0]  # Disregard batch and channel dimension

    # Post-process and save output (to specified location)
    output = (np.clip(output.detach().cpu().numpy(), 0, 1) * 255).astype(
        np.uint8)  # Clip to range [0, 1] and revert normalization (* 255)
    output = Image.fromarray(output)
    output = output.convert("RGB")
    output.save(path_out)

# Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the AOSNet model on a set of focal plane images.')
    parser.add_argument('weights_path', type=str, help='Path to the model weights.')
    parser.add_argument('focal_plane_1_path', type=str, help='Path to the focal plane image at 0.1m above ground.')
    parser.add_argument('focal_plane_2_path', type=str, help='Path to the focal plane image at 0.4m above ground.')
    parser.add_argument('focal_plane_3_path', type=str, help='Path to the focal plane image at 1.5m above ground.')
    parser.add_argument('path_out', type=str, help='Path for saving the output image.')

    args = parser.parse_args()

    main(args.weights_path, args.focal_plane_1_path, args.focal_plane_2_path, args.focal_plane_3_path, args.path_out)
