import cv2
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from camera_calibration.undistort_image import Undistort
import pickle
from line_finder.transform_perspective import TransformPerspective


def generate_matrix(src, dst):
    """
    generate perspective matrix and inverse perspective matrix
    :param src: source vertices
    :param dst: destine vertices, coordinates of source vertices on transformed img
    :return: perspective matrix and inverse perspective matrix
    """
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def main():
    perspective_matrix_path = cfg.line_finder['perspective_matrix_file']

    height = 720
    src = np.float32([[195, height], [593, 450], [689, 450], [1125, height]])
    dst = np.float32([[315, height], [315, 0], [965, 0], [965, height]])
    M, Minv = generate_matrix(src, dst)
    dict_pickle = dict()
    dict_pickle['M'] = M
    dict_pickle['Minv'] = Minv
    pickle.dump(dict_pickle, open(perspective_matrix_path, "wb"))

    transformer = TransformPerspective()

    image_name = 'straight_lines1'

    image_path = cfg.join_path(cfg.line_finder['input'], image_name + '.jpg')
    warped_img_path = cfg.join_path(cfg.line_finder['output'], image_name + '_warped.jpg')
    warped_unwarped_path = cfg.join_path(cfg.line_finder['output'], image_name + '_warped_unwarped.jpg')

    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    undistort = Undistort()
    img = undistort.undistort_image(img)

    warped_img = transformer.transform(img)

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

    warped_unwarped_img = transformer.inverse_transform(warped_img)
    warped_unwarped_img = cv2.cvtColor(warped_unwarped_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(warped_unwarped_path, warped_unwarped_img)

if __name__ == '__main__':
    main()
