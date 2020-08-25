import numpy as np


def normalise_sss_img(waterfall_image, clip_max=5):
    """Given a sss_waterfall_image from draping, process the
    image and return a normalised version where the column-wise
    mean is set to 1."""

    img_mean = waterfall_image.mean(axis=0)

    # set points around nadir to 0
    waterfall_image = waterfall_image.copy()
    waterfall_image[:, img_mean < 1e-1] = 0

    img_normalised = np.divide(
        waterfall_image,
        img_mean,
        out=np.zeros_like(waterfall_image),
        where=(img_mean != 0))
    img_normalised = np.clip(a=img_normalised, a_min=img_normalised.min(), a_max=clip_max)

    return img_normalised
