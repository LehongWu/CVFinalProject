import torch
from pytorch_fid import fid_score
import torchvision
# FID
def get_fid_score(real_folder: str, generated_folder: str, batch_size=32, dims=64):
    # both folders should include images files (.png, .jpg, etc)
    # size, name, format of these images don't have to be same
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    fid_value = fid_score.calculate_fid_given_paths([real_folder, generated_folder], batch_size=batch_size, device=device, dims=dims)
    return fid_value