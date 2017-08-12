import cv2
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from camera_calibration.undistort_image import Undistort


# todo: save M, check undistort, is that good?
def unwarp_image(img, src, dst):
    """
    warp an chessboard image
    :param img: input image
    :param row_number: number of rows
    :param col_number: number of columns
    :param offset: offset of the four outmost inner corners in the warped image
    :return: warped image and perspective matrix
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_size = (gray.shape[1], gray.shape[0])  # `img_size = gray.shape` should also work

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M


def main():
    image_name = 'straight_lines1'

    image_path = cfg.join_path(cfg.line_finder['input'], image_name + '.jpg')
    warped_img_path = cfg.join_path(cfg.line_finder['output'], image_name + '_warped.jpg')

    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    undistort = Undistort()
    img = undistort.undistort_image(img)

    height, width = img.shape[0], img.shape[1]
    src = np.float32([[195, height], [593, 450], [687, 450], [1125, height]])
    dst = np.float32([[315, height], [315, 0], [965, 0], [965, height]])

    warped_img, perspective_M = unwarp_image(img, src, dst)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.plot(src[:, 0], src[:, 1], '-r')
    ax1.set_title('Original Image', fontsize=50)

    ax2.imshow(warped_img)
    ax2.plot(dst[:, 0], dst[:, 1], '-r')
    ax2.set_title('Warped Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(warped_img_path)


if __name__ == '__main__':
    main()
