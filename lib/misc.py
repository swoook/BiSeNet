import numpy as np
import cv2


def letterbox_resize(src, dst_h, dst_w):
    # https://stackoverflow.com/a/48450206
    # https://github.com/qqwweee/keras-yolo3/issues/330#issue-401027524
    """Resize with same aspect ratio as source image.

    Args:
        src: `np.array`, source image.
        dst_h: `int`, height of target image.
        dst_w: `int`, width of target image.

    Returns:
        `np.array`, target image with same aspect ratio as source image.
    """
    dst = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    src_h, src_w = src.shape[:2]
    scale_factor = min(dst_h/src_h, dst_w/src_w)
    inter_h, inter_w = int(src_h * scale_factor), int(src_w * scale_factor)
    resized = cv2.resize(src, (inter_w, inter_h))
    top_idx = (dst_h-inter_h)//2
    left_idx = (dst_w-inter_w)//2
    dst[top_idx:top_idx+inter_h, left_idx:left_idx+inter_w, :] = resized
    return dst