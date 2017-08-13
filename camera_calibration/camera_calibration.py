import pickle
import cv2
import numpy as np
import glob
import config as cfg


def find_corners(image_paths, image_output_folder, row_number, col_number,
                 obj_points_filename=None, img_points_filename=None):
    """
    find chess board corners of images whose paths are given, save images with corner drawn in the given output folder
    also save the object points and image points in npy files
    :param image_paths: paths of images containing chess board
    :param image_output_folder: folder to save images with corner drawn
    :param row_number: number of rows of inner corners on the chess board
    :param col_number: number of columns of inner corners on the chess board
    :param obj_points_filename: filename to save the object points
    :param img_points_filename: filename to save the image points
    :return: object points and image points
    """

    # Arrays to store object points and image points from ALL the images.
    obj_points = []  # 3d points in real world space
    img_points = []  # 2d points in image plane.

    # prepare object points for a SINGLE image, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    obj_point = np.zeros((row_number * col_number, 3), np.float32)
    obj_point[:, :2] = np.mgrid[0:col_number, 0:row_number].T.reshape(-1, 2)

    # Step through the list of calibration images and search for chessboard corners
    for idx, file_name in enumerate(image_paths):

        if 'jpg' in file_name:
            img = cv2.imread(file_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (col_number, row_number), None)

            # If found, add object points, image points
            if ret:
                obj_points.append(obj_point)
                img_points.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (col_number, row_number), corners, ret)
                write_name = cfg.join_path(image_output_folder, file_name.split('/')[-1])
                cv2.imwrite(write_name, img)
                # cv2.imshow('img', img)
                # cv2.waitKey(500)

    # cv2.destroyAllWindows()
    if obj_points_filename is not None and img_points_filename is not None:
        np.save(obj_points_filename, obj_points)
        np.save(img_points_filename, img_points)
    return obj_points, img_points


def camera_calibration(obj_points, img_points, img_size, camera_calibration_file=None):
    """
    calibrate camera, and save the calibration result in pickle file
    :param obj_points: object points
    :param img_points: image points
    :param img_size: image size
    :param camera_calibration_file: camera calibration file
    :return: camera matrix and distortion coefficient from calibration
    """
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    if camera_calibration_file:
        dist_pickle = dict()
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump(dist_pickle, open(camera_calibration_file, "wb"))
    return [mtx, dist]


def main():

    input_path = cfg.camera_calibration['input']

    if input_path is not None:
        image_paths = glob.glob(cfg.join_path(input_path, 'calibration*.jpg'))

        obj_points, img_points = find_corners(image_paths, cfg.camera_calibration['output'],
                                              cfg.camera_calibration['grid_rows'],
                                              cfg.camera_calibration['grid_columns'])

        image_name = 'calibration8'
        image_path = cfg.join_path(input_path, image_name + '.jpg')
        img = cv2.imread(image_path)
        img_size = (img.shape[1], img.shape[0])

        camera_calibration(obj_points, img_points, img_size, cfg.camera_calibration['camera_calibration_file'])


if __name__ == '__main__':
    main()
