from utils.common_functions import *
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from utils.filters import *

import math

class BlockSegmentation:
    def __init__(self, image, lang="en", h_coeff=None):
        self.image = image
        if h_coeff is None:
            self.h_coeff = 0.5
        else:
            self.h_coeff = h_coeff
        if lang == "en":
            self.block_size = 256
        else:
            self.block_size = 128
        # self.block_size = 256


    def segment(self):
        filters = Filters()

        # image = remove_shadow(image)

        imageGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Noise removal with gaussian
        imageGray = gaussian(imageGray, 1)

        # Thresholding
        imageGray *= 255
        threshold = np.round(threshold_otsu(imageGray))
        # threshold = np.round(filters.otsu_segmentation(imageGray))
        imageGray[(imageGray > threshold)] = 255
        imageGray[(imageGray <= threshold)] = 0

        top, bottom = extract_text(imageGray)
        imageGray = imageGray[top:bottom, :]

        blocks = self.fill_blocks(bounding_rects=getBoundingRects(imageGray))

        return blocks

    # Segment the paper into blocks, and removing white space between words
    def fill_blocks(self, bounding_rects):
        image_shape = self.image.shape
        small_components_ratio = 375 / 8780618

        # initialize variables needed
        # block to save one block, when it is full we clear it
        block = np.full((self.block_size, self.block_size), 1, dtype='int')

        # heights to save the height of the bounding rect to calculate average of heights for new line
        heights = np.asarray([])

        # blocks to save the resultant blocks
        blocks = []

        # x, variable that identifies the current column (current width)
        x = 0

        # y, variable that identifies the current row (current height)
        y = 0

        size = len(bounding_rects)
        index = 0
        while index < size:
            bounding_rect = bounding_rects[index]

            if bounding_rect.height * bounding_rect.width <= small_components_ratio * (image_shape[0] * image_shape[1]):
                index += 1
                continue

            # show_images([np.multiply(bounding_rect.rect, 255)])

            if x == self.block_size:
                x = 0
                y += int(math.ceil((np.average(heights)) * self.h_coeff))
                heights = np.asarray([])

            if y == self.block_size:
                # New Block
                x = 0
                y = 0
                blocks.append(block)
                block = np.full((self.block_size, self.block_size), 1, dtype='int')
                heights = np.asarray([])

            if x + bounding_rect.width > self.block_size:
                if y + bounding_rect.height > self.block_size:
                    block[y:self.block_size - 1, x:self.block_size - 1] = np.multiply(
                        block[y:self.block_size - 1, x:self.block_size - 1],
                        bounding_rect.rect[0:self.block_size - 1 - y, 0:self.block_size - 1 - x])
                    # show_images([np.multiply(block, 255)])
                    new_bounding_rect = BoundingRect(bounding_rect.height - (self.block_size - 1 - y),
                                                     bounding_rect.width - (self.block_size - 1 - x),
                                                     bounding_rect.rect[self.block_size - 1 - y:,
                                                     self.block_size - 1 - x:])
                    bounding_rects = np.insert(bounding_rects, index + 1, new_bounding_rect)
                    size += 1

                    # New Block
                    x = 0
                    y = 0
                    # cv2.imwrite('blocksample'+str(index)+'.png', np.multiply(block, 255))
                    blocks.append(block)
                    block = np.full((self.block_size, self.block_size), 1, dtype='int')
                    heights = np.asarray([])

                else:
                    block[y:y + bounding_rect.height, x:self.block_size - 1] = np.multiply(
                        block[y:y + bounding_rect.height, x:self.block_size - 1],
                        bounding_rect.rect[:, 0:self.block_size - 1 - x])
                    # show_images([np.multiply(block, 255)])
                    heights = np.append(heights, bounding_rect.height)
                    new_bounding_rect = BoundingRect(bounding_rect.height,
                                                     bounding_rect.width - (self.block_size - 1 - x),
                                                     bounding_rect.rect[:, self.block_size - 1 - x:])
                    bounding_rects = np.insert(bounding_rects, index + 1, new_bounding_rect)
                    size += 1

                    x = 0
                    y += int((np.average(heights)) / 2)
                    heights = np.asarray([])

            else:
                if y + bounding_rect.height > self.block_size:
                    block[y:self.block_size - 1, x:x + bounding_rect.width] = np.multiply(
                        block[y:self.block_size - 1, x:x + bounding_rect.width],
                        bounding_rect.rect[0:self.block_size - 1 - y, :])
                    # show_images([np.multiply(block, 255)])

                    x += bounding_rect.width
                    heights = np.append(heights, self.block_size - y)

                    new_bounding_rect = BoundingRect(bounding_rect.height - (self.block_size - 1 - y),
                                                     bounding_rect.width,
                                                     bounding_rect.rect[self.block_size - 1 - y:, :])
                    bounding_rects = np.insert(bounding_rects, index + 1, new_bounding_rect)
                    size += 1
                else:
                    block[y:y + bounding_rect.height, x:x + bounding_rect.width] = np.multiply(
                        block[y:y + bounding_rect.height, x:x + bounding_rect.width], bounding_rect.rect)
                    # show_images([np.multiply(block, 255)])
                    x += bounding_rect.width
                    heights = np.append(heights, bounding_rect.height)

            index += 1
        if y < self.block_size:
            blocks.append(block)
        return blocks
