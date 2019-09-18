import cv2
import numpy as np
from skimage.filters import gaussian


def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(image, parsing, color=(230, 50, 20)):
    parsing = parsing.argmax(0)
    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

    color = np.array(color, dtype='uint8')
    color.shape = 1, 1, 3

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

    image_hsv[..., 0] = tar_hsv[..., 0]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    changed = sharpen(changed)

    changed[parsing != 17] = image[parsing != 17]
    return changed


if __name__ == '__main__':
    parsing = np.load('parsing.npy')
    image = cv2.imread('makeup/116_ori.png')

    color = [100, 200, 100]
    image = hair(image, parsing, color)
    cv2.imwrite('116_2.jpg', image)
