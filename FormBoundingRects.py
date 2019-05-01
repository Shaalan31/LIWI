import cv2
import numpy as np
from BoundingRect import BoundingRect
from commonfunctions import *


def getBoundingRects(image, image_shape):
    image = image.astype('uint8')
    # show_images([image])
    # finding contours whose parent is the bounding rect of the whole paper
    # since hierarchy[:, 3] gives us the id of the parent

    small_components_ratio = 375 / 8780618

    all_bounding_rects = np.asarray([])
    contours, hierarchy = cv2.findContours(np.subtract(255, image.copy()), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # hierarchy = hierarchy[0]
    contours = np.asarray(contours)
    # mask = (hierarchy[:, 3] == -1).astype('int')
    # contours = contours[np.where(mask)]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if int(w * h) > small_components_ratio * (image_shape[0] * image_shape[1]):
            # we need to discard this bounding rect since it's not logical
            if h > 256:
                continue
            new_bounding_rect = BoundingRect(h, w, np.divide(image[y:y + h, x:x + w], 255))
            all_bounding_rects = np.append(all_bounding_rects, new_bounding_rect)

    return all_bounding_rects
