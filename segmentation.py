from commonfunctions import *
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from FormBoundingRects import *


def segment(image):
    image = remove_shadow(image)

    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise removal with gaussian
    imageGray = gaussian(imageGray, 1)

    # Thresholding
    imageGray *= 255
    threshold = np.round(threshold_otsu(imageGray) * 1.1)
    imageGray[(imageGray > threshold)] = 255
    imageGray[(imageGray <= threshold)] = 0

    # imageGray = cv2.erode(imageGray.copy(), np.ones((3, 3)), iterations=1)

    # show_images([imageGray])
    top, bottom = extract_text(imageGray)
    imageGray = imageGray[top:bottom, :]
    # show_images([imageGray])

    blocks = fill_blocks(getBoundingRects(imageGray, imageGray.shape))
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
def fill_blocks(bounding_rects):
    # initialize variables needed
    # block to save one block, when it is full we clear it
    block = np.ones((256, 256), dtype='int')

    # heights to save the height of the bounding rect to calculate average of heights for new line
    heights = np.asarray([])

    # blocks to save the resultant blocks
    blocks = np.asarray([])

    # x, variable that identifies the current column (current width)
    x = 0

    # y, variable that identifies the current row (current height)
    y = 0

    # loop for processing each bounding rect
    for index, bounding_rect in enumerate(bounding_rects, 1):
        if np.max(bounding_rect.rect) > 1:
            bounding_rect.rect = 1 - np.divide(bounding_rect.rect, 255).astype('int')
        else:
            bounding_rect.rect = 1 - bounding_rect.rect
        # Check if the height greater than 256
        # move to another block, and clear the variables
        if y + bounding_rect.height > 256:
            y = 0
            x = 0
            heights = np.asarray([])
            block = 1 - block
            show_images([block])
            blocks = np.append(blocks, block)
            block = np.ones((256, 256), dtype='int')
            block[y:y + bounding_rect.height - 1, x:x + bounding_rect.width - 1] = np.multiply(
                block[y:y + bounding_rect.height - 1, x:x + bounding_rect.width - 1], bounding_rect.rect)
        else:
            # filling the block with the bounding rect & incrementing the width x
            # & append the height of the bounding rect to the heights
            block[y:y + bounding_rect.height - 1, x:x + bounding_rect.width - 1] = np.multiply(
                block[y:y + bounding_rect.height - 1, x:x + bounding_rect.width - 1], bounding_rect.rect)
            x += bounding_rect.width
            heights = np.append(heights, bounding_rect.height)

            # getting next bounding rect and check if it will fit in the remaining block,
            # if not will split the bounding rect into two bounding rects, one will fill the remaining part
            # the other will replace the next bounding rect, and increment the height y and clearing heights & width
            if index + 1 < bounding_rects.shape[0]:
                next_bounding_rect = bounding_rects[index + 1]
                next_bounding_rect.rect = 1 - np.divide(next_bounding_rect.rect, 255).astype('int')

                if x + next_bounding_rect.width - 1 > 256:
                    block[y:y + next_bounding_rect.height - 1, x:255] = np.multiply(
                        block[y:y + next_bounding_rect.height - 1, x:255], next_bounding_rect.rect[:, 0:255 - x])
                    heights = np.append(heights, next_bounding_rect.height)
                    y += int((np.average(heights)) / 2)
                    heights = np.asarray([])
                    bounding_rects[index + 1].rect = 1 - next_bounding_rect.rect[:, 256 - x:]
                    bounding_rects[index + 1].width = next_bounding_rect.width - (256 - x)
                    x = 0

                    # if the height is greater than 256, then will start to fill new block,
                    # and append the filled block to blocks, and setting the height y to 0
                    if y > 256:
                        block = 1 - block
                        show_images([block])
                        blocks = np.append(blocks, block)
                        block = np.ones((256, 256), dtype='int')
                        y = 0
    return blocks
