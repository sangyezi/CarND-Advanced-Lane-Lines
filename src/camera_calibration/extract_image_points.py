import numpy as np
import cv2
import glob
import os


def find_corners(image_paths, image_output_folder, row_number, col_number, obj_points_filename, img_points_filename):
    """
    find chess board corners of images whose paths are given, save images with corner drawn in the given output folder
    also save the object points and image points in npy files
    :param image_paths: paths of images containing chess board
    :param image_output_folder: folder to save images with corner drawn
    :param row_number: number of rows of inner corners on the chess board
    :param col_number: number of columns of inner corners on the chess board
    :param obj_points_filename: filename to save the object points
    :param img_points_filename: filename to save the image points
    :return: None
    """

    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)

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
                write_name = os.path.join(image_output_folder, file_name.split('/')[-1])
                cv2.imwrite(write_name, img)
                # cv2.imshow('img', img)
                # cv2.waitKey(500)

    # cv2.destroyAllWindows()

    np.save(obj_points_filename, obj_points)
    np.save(img_points_filename, img_points)


def main():
    grid_rows = 6
    grid_columns = 9
    base_dir = os.path.dirname(__file__)

    image_folder = os.path.abspath(os.path.join(base_dir, '..', '..', 'camera_cal/calibration*.jpg'))
    image_paths = glob.glob(image_folder)

    image_output_folder = os.path.abspath(os.path.join(base_dir, '..', '..', 'camera_cal_corners'))

    obj_points_file = "obj_points.npy"
    img_points_file = "img_points.npy"

    find_corners(image_paths, image_output_folder, grid_rows, grid_columns, obj_points_file, img_points_file)

if __name__ == '__main__':
    main()
