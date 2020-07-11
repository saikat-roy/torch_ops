import torch


def _patch_nd(volume, p_len, p_step):

    d = len(volume.shape) - 2  # Subtract the batch and channel dimensions
    b, c = volume.shape[0], volume.shape[1]
    if isinstance(p_len, int):
        p_len = [p_len for _ in range(d)]
    if isinstance(p_step, int):
        p_step = [p_step for _ in range(d)]

    reshaped = [int((volume.shape[idx+2]-(p_len[idx]-p_step[idx]))
                    /(p_len[idx]-(p_len[idx]-p_step[idx])))
                for idx in range(d)]

    assert len(p_len) == d  # Integrity check if manually passing p_len list
    assert len(p_step) == d # Integrity check if manually passing p_len list

    for idx in range(d):
        volume = volume.unfold(idx+2, p_len[idx], p_step[idx])
    try:
        volume = volume.reshape(-1, c, *reshaped)
    except:
        print("Reshape attempt failed. Please reshape manually")
    return volume


def extract_patches(volume, p_len, p_step):
    """
    Extracts patches from torch tensors. Will attempt simple resizing after unfolding (torch.unfold) to extract patches
    but returns un-reshaped tensor after unfolding  on fail. Patches are stacked along batch dimension.

    :param volume: n-d Torch tensor of shape (b, c, d1, ..., dn). Eg. for 2D, vol.shape = torch.Size([b, c, d1, d2])
    :param p_len: int or list, LENGTH of the patches
    :param p_step: int or list, STEP SIZE for the patching filter
    :return: torch.Tensor
    """
    assert isinstance(p_step, int) or isinstance(p_step, list)
    assert isinstance(p_len, int) or isinstance(p_len, list)
    return _patch_nd(volume, p_len, p_step)


# if __name__ == "__main__":
#
#     vol = torch.rand((1,4,64,64,64))
#     p = extract_patches(vol, [4,4,4], 2)
#     print(p.shape)