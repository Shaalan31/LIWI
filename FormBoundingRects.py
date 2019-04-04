import cv2
import numpy as np
from BoundingRect import BoundingRect


def getBoundingRects(image, image_shape):
    example = image.astype('uint8')
    example_copy = example.copy()

    # finding contours whose parent is the bounding rect of the whole paper
    # since hierarchy[:, 3] gives us the id of the parent

    small_components_ratio = 375 / 8780618

    all_bounding_rects = np.asarray([])

    _, contours, hierarchy = cv2.findContours(example_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]
    contours = np.asarray(contours)
    mask = (hierarchy[:, 3] == 0).astype('int')
    contours = contours[np.where(mask)]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if int(w * h) > small_components_ratio * (image_shape[0] * image_shape[1]):
            new_bounding_rect = BoundingRect(h, w, image[y:y + h - 1, x:x + w - 1])
            all_bounding_rects = np.append(all_bounding_rects, new_bounding_rect)

    return all_bounding_rects
