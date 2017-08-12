import pickle
import cv2
import matplotlib.pyplot as plt
import config as cfg


class Undistort:
    """
    Provide functionality to undistort images from a camera whose matrix and distort coefficient are available
    """
    def __init__(self):
        """
        initiation, read camera matrix and distort coefficient from disk
        """
        camera_pickle_path = cfg.camera_calibration['pickle_filename']

        try:
            dist_pickle = pickle.load(open(camera_pickle_path, "rb"))
            self.mtx = dist_pickle["mtx"]
            self.dist = dist_pickle["dist"]
        except Exception:
            self.mtx = None
            self.dist = None

    def undistort_image(self, img):
        """
        undistort a image
        raise ValueError if camera calibration are not available
        :param img: image to be undistroted
        :return: undistroted image
        """
        if self.mtx is not None and self.dist is not None:
            return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        else:
            raise ValueError("unable to process the image, camera data not available")

    def plot_contrast(self, original, undistorted, save_path=None):
        """
        plot the original and undistorted images together, show the image if save_path is not provided
        save the image if save path is provided
        :param original: original image
        :param undistorted: undistorted image
        :param save_path:  path to save the image
        :return: None
        """
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        undistorted = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(original)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(undistorted)
        ax2.set_title('Undistorted Image', fontsize=30)
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()


def main():
    undistort = Undistort()

    image_name = 'calibration10'
    image_path = cfg.join_path(cfg.camera_calibration['input'], image_name + '.jpg')

    img = cv2.imread(image_path)

    undistorted_img_save_path = cfg.join_path(cfg.camera_calibration['output'], image_name + '_undistorted.jpg')

    contrast_images_save_path = cfg.join_path(cfg.camera_calibration['output'], image_name + '_contrast.jpg')

    undistorted_img = undistort.undistort_image(img)
    cv2.imwrite(undistorted_img_save_path, undistorted_img)

    undistort.plot_contrast(img, undistorted_img, contrast_images_save_path)


if __name__ == '__main__':
    main()
