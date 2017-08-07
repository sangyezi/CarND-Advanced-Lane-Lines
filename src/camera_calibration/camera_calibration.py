import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def camera_calibration(obj_points, img_points, img_size, pickle_filename):
    """
    calibrate camera, and save the calibration result in pickle file
    :param obj_points: object points
    :param img_points: image points
    :param img_size: image size
    :param pickle_filename: pickle file name
    :return: camera matrix and distortion coefficient from calibration
    """
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = dict()
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open(pickle_filename, "wb"))
    return [mtx, dist]


def undistort_image(mtx, dist, image_path, output_path, contrast_path):
    """
    undistort an given image, save the undistorted image, and save a contrast plot containing the two images
    :param mtx: camera matrix from calibration
    :param dist: distortion coefficient from calibration
    :param image_path: path of the image to be undistorted
    :param output_path: path to save the undistorted image
    :param contrast_path: path to save the contrast plot
    :return: None
    """
    img = cv2.imread(image_path)

    dst = cv2.undistort(img, mtx, dist, None, mtx)

    cv2.imwrite(output_path, dst)

    # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.savefig(contrast_path)


def main():
    obj_points_filename = "obj_points.npy"
    img_points_filename = "img_points.npy"
    try:
        obj_points = np.load(obj_points_filename)
        img_points = np.load(img_points_filename)
    except FileNotFoundError:
        print("use find_corners in extract_image_points to save object and image points first")
        return

    base_dir = os.path.dirname(__file__)

    image_name = 'calibration8'
    image_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'camera_cal', image_name + '.jpg'))
    img = cv2.imread(image_path)
    img_size = (img.shape[1], img.shape[0])

    pickle_filename = "wide_dist_pickle.p"

    [mtx, dist] = camera_calibration(obj_points, img_points, img_size,  pickle_filename)

    output_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'camera_cal_corners', image_name + '_undistorted.jpg'))
    contrast_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'camera_cal_corners', image_name + '_contrast.jpg'))

    undistort_image(mtx, dist, image_path, output_path, contrast_path)

if __name__ == '__main__':
    main()
