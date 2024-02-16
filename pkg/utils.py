from typing import List
import numpy as np
from matplotlib import pyplot as plt


def mask_to_rle(mask: np.ndarray):
    """
    Convert a binary mask to RLE format.
    :param mask: numpy array, 1 - mask, 0 - background
    :return: RLE array
    """
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return [int(x) for x in runs]

def rle_to_mask(rle: List[int], shape: tuple):
    """
    Convert RLE string to mask
    :param rle: run-length as string formated (start length)
    :param shape: (height, width) of array to return
    :return:
    """
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if len(rle) == 1:
        return mask.reshape(shape).T
    for i, start_pixel in enumerate(rle[::2]):
        mask[int(start_pixel): int(start_pixel + rle[2 * i + 1])] = 1
    return mask.reshape(shape).T

def mask_to_rgb(mask: np.ndarray, label_to_color: dict):
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    mask_red = np.zeros_like(mask, dtype=np.uint8)
    mask_green = np.zeros_like(mask, dtype=np.uint8)
    mask_blue = np.zeros_like(mask, dtype=np.uint8)

    for l in label_to_color:
        mask_red[mask == l] = label_to_color[l][0]
        mask_green[mask == l] = label_to_color[l][1]
        mask_blue[mask == l] = label_to_color[l][2]

    mask_colors = (
        np.stack([mask_red, mask_green, mask_blue]).astype(np.uint8).transpose(1, 2, 0)
    )
    return mask_colors


def show_in_grid(
    images: List[np.ndarray],
    num_rows,
    num_cols,
    show_plot=False,
    savefig_path=None,
    size_factor=10,
    x_labels=[],
):
    plt.ioff()
    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * size_factor, num_rows * size_factor)
    )
    fig.tight_layout()

    for i, img in enumerate(images):
        row = i // num_cols
        col = i % num_cols
        if num_rows > 1:
            ax = axs[row, col]
        else:
            ax = axs[i]
        ax.axis("off")
        ax.imshow(img)

        if row == num_rows - 1:
            if len(x_labels) > 0:
                ax.set_title(x_labels[col])

    if show_plot:
        plt.show()
    else:
        fig.savefig(savefig_path)
        plt.close()
