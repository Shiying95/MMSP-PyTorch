from .pixelbert import (
    pixelbert_transform,
    pixelbert_transform_randaug,
    pixelbert_transform_nonresize,
)

_transforms = {
    "pixelbert": pixelbert_transform,
    "pixelbert_randaug": pixelbert_transform_randaug,
    "pixelbert_nonresize": pixelbert_transform_nonresize,
}


def keys_to_transforms(keys: list, size=224):
    return [_transforms[key](size=size) for key in keys]
