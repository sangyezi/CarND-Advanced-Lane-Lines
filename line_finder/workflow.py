from camera_calibration.undistort_image import Undistort
from line_finder.transform_perspective import TransformPerspective

from line_finder.thresholding import threshold_pipeline
from line_finder.locate_lane_lines import Locator, LocatorWithPrior

import cv2
import numpy as np
import config as cfg
import matplotlib.pyplot as plt


class Lane:
    """
    Lane on a image, composed of left lane line and right lane line
    """
    def __init__(self, img):
        """
        initialize lane on a image without prior knowledge about the lane
        :param img: input image (in BGR color)
        """
        self.left_lane = Line()
        self.right_lane = Line()

        self.undistorter = Undistort()
        self.transformer = TransformPerspective()

        self.undist = self.undistorter.undistort_image(img)
        thresholded, thresholded_masked = threshold_pipeline(self.undist, False)
        self.warped = self.transformer.transform(thresholded)

        locator = Locator(self.warped)
        left_located_line, right_located_line = locator.sliding_window()
        self.left_lane.add_fit(left_located_line)
        self.right_lane.add_fit(right_located_line)

        self.locator_visualized = locator.visualize()

        # "update" based on located line on the current image
        self.update(img)

    def update(self, img):
        """
        identify lane on current image, using lane identified from previous images (previous frames of a series) as prior knowledge
        :param img: current image as input
        :return: None
        """
        self.undist = self.undistorter.undistort_image(img)
        thresholded, thresholded_masked = threshold_pipeline(self.undist, False)
        self.warped = self.transformer.transform(thresholded)

        locatorWithPrior = LocatorWithPrior(self.warped, self.left_lane.fit, self.right_lane.fit)
        left_located_line, right_located_line = locatorWithPrior.sliding_window()
        self.left_lane.add_fit(left_located_line)
        self.right_lane.add_fit(right_located_line)
        # todo: consider return some value based on add_fit: if locatorWithPrior does not work well, use locator instead

        self.locator_with_prior_visualized = locatorWithPrior.visualize()

    def visualize(self, panel_scale=0.3):
        """
        generate output image of undistorted original image and identified lane line,
        with a small panel displaying identified lane line on the top view perspective of undistorted original image
        :param panel_scale: scale of the panel on the output image
        :return: output image
        """
        panel_img = cv2.resize(self.locator_with_prior_visualized, (0, 0), fx=panel_scale, fy=panel_scale)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(self.warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        ploty, left_fitx, right_fitx = Locator.pixels_on_fit(self.warped.shape[0], self.left_lane.fit, self.right_lane.fit)

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.transformer.inverse_transform(color_warp)

        result = cv2.addWeighted(self.undist, 1, newwarp, 0.3, 0)

        result = cv2_overlay(result, panel_img, (5, 5))
        return result


class Line:
    def __init__(self, N=10):
        # number of previous fit to cache
        self.N = N
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last N fits of the line
        self.recent_fitx = []
        # average x values of the fitted line over the last N iterations
        self.average_fitx = None

        # y value of the fitted line
        self.fity = None

        # polynomial coefficients of the last N fits of the line
        self.recent_fit_coefficients = []
        # polynomial coefficients averaged over the last N iterations
        self.average_fit_coefficients = None
        # fit on average x
        self.fit_on_average = None
        # best fit, could be either most recent fit, average_fit or fit_on_average
        self.fit = None

        # x values for detected line pixels
        self.line_x = None
        # y values for detected line pixels
        self.line_y = None

        # todo: deal with the fields below
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

    def update_average_fitx(self, confidence=None):
        if confidence is None:
            confidence = 1 / self.N
        if self.average_fitx is None:
            self.average_fitx = self.recent_fitx[-1]
        else:
            self.average_fitx = self.average_fitx * (1 - confidence) + self.recent_fitx[-1] * confidence

    def update_average_fit(self, confidence=None):
        if confidence is None:
            confidence = 1 / self.N
        if self.average_fit_coefficients is None:
            self.average_fit_coefficients = self.recent_fit_coefficients[-1]
        else:
            self.average_fit_coefficients = self.average_fit_coefficients * (1 - confidence) \
                                            + self.recent_fit_coefficients[-1] * confidence

    def add_fit(self, located_line):
        # todo: calculate confidence of located_line based on parallelism, diff from previous lines etc

        self.detected = True
        self.recent_fitx.append(located_line.fitx)
        if len(self.recent_fitx) > self.N:
            self.recent_fitx.pop(0)
        self.update_average_fitx()

        self.fity = located_line.fity

        self.recent_fit_coefficients.append(located_line.fit_coefficients)
        self.update_average_fit()
        self.fit_on_average = np.polyfit(self.average_fitx, self.fity, 2)

        # currently use most recent fit as fit
        self.fit = self.recent_fit_coefficients[-1]

        self.line_x = located_line.line_x
        self.line_y = located_line.line_y


def cv2_overlay(image, panel, offset):
    alpha_s = 0.7
    alpha_l = 1.0 - alpha_s

    x1, x2 = offset[0], offset[0] + panel.shape[1]
    y1, y2 = offset[1], offset[1] + panel.shape[0]

    # image[y1:y2, x1:x2, :] = panel
    for c in range(0, 3):
        image[y1:y2, x1:x2, c] = (alpha_s * panel[:, :, c] +
                                  alpha_l * image[y1:y2, x1:x2, c])
    cv2.rectangle(image, (x1, y1), (x2, y2), (192, 192, 192), 3)
    return image


def test():
    """
    test the workflow using a test image
    :return:
    """
    image_name = 'test6'
    image_path = cfg.join_path(cfg.line_finder['input'], image_name + '.jpg')
    image = cv2.imread(image_path)
    lane = Lane(image)
    result = lane.visualize()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    plt.imshow(result)
    output_path = cfg.join_path(cfg.line_finder['output'], image_name + '_line.jpg')
    plt.show()
    # plt.savefig(output_path)


def process_image(image):
    """
    process image (identify the lane line), return the processed image
    :param image: input image
    :return: the processed image
    """
    lane = Lane(image)
    return lane.visualize()

if __name__ == '__main__':
    # test()

    from moviepy.editor import VideoFileClip

    video_name = 'project_video.mp4'

    input_path = cfg.join_path(cfg.video_path['videos'], video_name)
    output_path = cfg.join_path(cfg.video_path['output_videos'], video_name)

    ## To speed up the testing process, only process a subclip of the first 5 seconds
    # clip1 = VideoFileClip(input_path).subclip(0, 5)
    clip1 = VideoFileClip(input_path)
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(output_path, audio=False)
