import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def unwarp_chessboard(img, row_number, col_number, offset):
    """
    warp an chessboard image
    :param img: input image
    :param row_number: number of rows
    :param col_number: number of columns
    :param offset: offset of the four outmost inner corners in the warped image
    :return: warped image and perspective matrix
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (col_number, row_number), None)

    src = np.float32([corners[0], corners[col_number - 1], corners[-1], corners[-col_number]])

    img_size = (gray.shape[1], gray.shape[0])  # `img_size = gray.shape` should also work

    dst = np.float32([[offset, offset],
                      [img_size[0] - offset, offset],
                      [img_size[0] - offset, img_size[1] - offset],
                      [offset, img_size[1] - offset]])

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M


def main():
    image_name = 'calibration8'
    base_dir = os.path.dirname(__file__)
    # img_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'camera_cal', image_name + '.jpg'))
    # note calibration2: undistorted image cannot find corners correctly, has to use the original image, as above

    img_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'camera_cal_corners', image_name + '_undistorted.jpg'))
    warped_img_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'camera_cal_corners', image_name + '_warped.jpg'))

    grid_rows = 6
    grid_columns = 9
    offset = 0

    img = cv2.imread(img_path)
    warped_img, perspective_M = unwarp_chessboard(img, grid_rows, grid_columns, offset)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(warped_img)
    ax2.set_title('Warped Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(warped_img_path)


if __name__ == '__main__':
    main()
