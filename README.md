# Torch Ops:
Random PyTorch ops that I happened to create over the years, sometimes with generous help from the PyTorch forums and Stackoverflow ofcourse.

Pretty useful until PyTorch adds official functions to its code base. 

## Currently Implemented:

1. Patch Extraction from n-D tensors. `utils.transforms.extract_patches` can extract patches from arbitrary n-D tensors (without breaking autograd) by specifying a filter shape and step.

## TODO:

1. Majority Voting downsampler for 2D and 3D tensors (Useful for downsampling segmentation label maps).
