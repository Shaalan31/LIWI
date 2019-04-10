from commonfunctions import *
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from FormBoundingRects import *


def segment(image):
    # image = remove_shadow(image)

    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise removal with gaussian
    imageGray = gaussian(imageGray, 1)

    # Thresholding
    imageGray *= 255
    threshold = np.round(threshold_otsu(imageGray))
    imageGray[(imageGray > threshold)] = 255
    imageGray[(imageGray <= threshold)] = 0

    # imageGray = cv2.erode(imageGray.copy(), np.ones((3, 3)), iterations=1)

    # show_images([imageGray])
    top, bottom = extract_text(imageGray)
    imageGray = imageGray[top:bottom, :]
    # show_images([imageGray])

    blocks = fill_blocks(getBoundingRects(imageGray, imageGray.shape), imageGray.shape)
    #
    # writer_lines = []
    # line_start = 0
    # foundALine = False
    # # imgName = 0
    # for line_index in range(imageGray.shape[0]):
    #     values, count = np.unique(imageGray[line_index, :], return_counts=True)
    #     if len(values) == 1:
    #         foundALine = False
    #         continue
    #     countBlack = count[0]
    #     countWhite = count[1]
    #     total = countWhite + countBlack
    #     percentageBlack = (countBlack / total) * 100
    #     if percentageBlack > 1 and not foundALine:
    #         foundALine = True
    #         line_start = line_index
    #     else:
    #         if foundALine and percentageBlack < 1:
    #             if line_index - line_start > (imageGray.shape[1] / 60):
    #                 line = cv2.copyMakeBorder(imageGray[line_start:line_index, :].astype('uint8'), 1, 1, 1, 1,
    #                                           cv2.BORDER_CONSTANT, value=[255, 255, 255])
    #                 # io.imsave('output/image' + str(imgName) + '.png', line)
    #                 # imgName += 1
    #                 # show_images([line])
    #                 writer_lines.append(line)
    #             foundALine = False
    return blocks


def extract_text(img):
    horizontal = np.copy(img)
    cols = horizontal.shape[1]
    horizontal_size = int(cols / 15)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = 255 - horizontal
    horizontal /= 255
    # show_images([horizontal])
    sum = np.sum(horizontal, axis=1)
    sum[sum < int(cols / 10)] = 0
    sum[sum > int(cols / 10)] = 1
    if np.max(sum) == np.min(sum):
        return 0, img.shape[0]
    half = int(sum.shape[0] / 2)
    top_boundary = half - np.argmax(sum[half:0:-1])
    bottom_boundary = half + np.argmax(sum[half:])

    return top_boundary + 2, bottom_boundary - 2


# Segment the paper into blocks, and removing white space between words
def fill_blocks(bounding_rects, image_shape):
    small_components_ratio = 375 / 8780618

    # initialize variables needed
    # block to save one block, when it is full we clear it
    block = np.full((256, 256), 1, dtype='int')

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

        if x == 256:
            x = 0
            y += int((np.average(heights)) / 2)
            heights = np.asarray([])

        if y == 256:
            # New Block
            x = 0
            y = 0
            blocks.append(block)
            block = np.full((256, 256), 1, dtype='int')
            heights = np.asarray([])

        if x + bounding_rect.width > 256:
            if y + bounding_rect.height > 256:
                block[y:255, x:255] = np.multiply(
                    block[y:255, x:255], bounding_rect.rect[0:255 - y, 0:255 - x])
                # show_images([np.multiply(block, 255)])
                new_bounding_rect = BoundingRect(bounding_rect.height - (255 - y), bounding_rect.width - (255 - x),
                                                 bounding_rect.rect[255 - y:, 255 - x:])
                bounding_rects = np.insert(bounding_rects, index + 1, new_bounding_rect)
                size += 1

                # New Block
                x = 0
                y = 0
                # show_images([np.multiply(block, 255)])
                blocks.append(block)
                block = np.full((256, 256), 1, dtype='int')
                heights = np.asarray([])

            else:
                block[y:y + bounding_rect.height, x:255] = np.multiply(
                    block[y:y + bounding_rect.height, x:255], bounding_rect.rect[:, 0:255 - x])
                # show_images([np.multiply(block, 255)])
                heights = np.append(heights, bounding_rect.height)
                new_bounding_rect = BoundingRect(bounding_rect.height, bounding_rect.width - (255 - x),
                                                 bounding_rect.rect[:, 255 - x:])
                bounding_rects = np.insert(bounding_rects, index + 1, new_bounding_rect)
                size += 1

                x = 0
                y += int((np.average(heights)) / 2)
                heights = np.asarray([])

        else:
            if y + bounding_rect.height > 256:
                block[y:255, x:x + bounding_rect.width] = np.multiply(
                    block[y:255, x:x + bounding_rect.width], bounding_rect.rect[0:255 - y, :])
                # show_images([np.multiply(block, 255)])

                x += bounding_rect.width
                heights = np.append(heights, 256 - y)

                new_bounding_rect = BoundingRect(bounding_rect.height - (255 - y), bounding_rect.width,
                                                 bounding_rect.rect[255 - y:, :])
                bounding_rects = np.insert(bounding_rects, index + 1, new_bounding_rect)
                size += 1
            else:
                block[y:y + bounding_rect.height, x:x + bounding_rect.width] = np.multiply(
                    block[y:y + bounding_rect.height, x:x + bounding_rect.width], bounding_rect.rect)
                # show_images([np.multiply(block, 255)])
                x += bounding_rect.width
                heights = np.append(heights, bounding_rect.height)

        index += 1

    return blocks
