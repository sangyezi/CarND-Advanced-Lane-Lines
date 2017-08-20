from camera_calibration.undistort_image import Undistort
from line_finder.transform_perspective import TransformPerspective

from line_finder.thresholding import threshold_pipeline, challenge_threshold_pipeline, challenge_threshold_pipeline_2
from line_finder.locate_lane_lines import Locator, LocatorWithPrior

import cv2
import numpy as np
import config as cfg
import matplotlib.pyplot as plt
import math


class Lane:
    """
    Lane on a image, composed of left lane line and right lane line
    """
    Y_M_PER_PIX = 30 / 720  # meters per pixel in y dimension
    X_M_PER_PIX = 3.7 / 700  # meters per pixel in x dimension

    def __init__(self, thresholder):
        """
        initialize lane on a image without prior knowledge about the lane
        :param img: input image (in BGR color)
        """
        self.left_lane = Line()
        self.right_lane = Line()

        # to undistort image
        self.undistorter = Undistort()
        # to transform perspective of the image to topview
        self.transformer = TransformPerspective()
        # threshold (use threshold to filter) image
        self.thresholder = thresholder

        # number of frame in hte video
        self.frame = 0

        # distance of center of image and center of two lane lines (in pixels)
        self.line_base_pos = None

        # image height in pixels
        self.height = 0
        # image weight in pixels
        self.width = 0
        # measurement of how parallel of the two lane lines
        self.parallelism = 1.0

        # undistorted image, in BGR channel
        self.undist = None
        # image after thresholding, single channel binary, used as the first panel in the final video
        self.thresholded = None
        # thresholded image after transform perspective to top view , single channel in binary
        self.warped = None
        # top view showing lane lines identified and fitted, used as the second panel in the final video
        self.locator_with_prior_visualized = None

        # average distance between left and right lane lines
        self.dist_between_lines = []
        self.average_dist_between_lines = 0

    def find_line(self, img):
        """
        identify the two lane lines on an input image, with or without prior knowledge
        :param img: image as input
        :return: None
        """
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.frame = self.frame + 1

        self.undist = self.undistorter.undistort_image(img)
        images = self.thresholder(self.undist)
        self.thresholded = images.get('thresholded_masked')
        self.warped = self.transformer.transform(self.thresholded)

        if not self.left_lane.detected or not self.right_lane.detected:
            self.fresh_start()
        # Notice: update alway happen, even after fresh_start, which could cause add_fit twice if refresh_start
        self.update()

        # if current lane lines are not to good, try to start from scratch again
        if not self.good_lane_lines():
            self.fresh_start()
            self.update()

        self.center_offset()

    def good_lane_lines(self):
        """
        if the identified lane lines are good, using two criteria:
        1. the left and right lines are relatively parallel
        2. the distance between left and right lines are not different from previous measurement
        :return:
        """
        dist_diff = np.abs(self.dist_between_lines[-1] - self.average_dist_between_lines) / self.average_dist_between_lines
        return self.parallelism >= 0.8 and dist_diff <= 0.2

    def fresh_start(self):
        """
        identify lane lines on current image, without prior knowledge
        :return:
        """
        # clear previously identified lane lines
        self.left_lane = Line()
        self.right_lane = Line()

        locator = Locator(self.warped)
        left_located_line, right_located_line = locator.sliding_window()

        self.left_lane.add_fit(left_located_line, 1)
        self.right_lane.add_fit(right_located_line, 1)
        # self.locator_visualized = locator.visualize()

    def update(self):
        """
        identify lane lines on current image, using previsouly identified lane lines as prior knowledge
        :return: None
        """
        locatorWithPrior = LocatorWithPrior(self.warped, self.left_lane.fit, self.right_lane.fit)
        left_located_line, right_located_line = locatorWithPrior.sliding_window()

        # calculate parallism, use as "confidence" about current fit for add_fit()
        self.get_parallelism(left_located_line, right_located_line)

        self.left_lane.add_fit(left_located_line, self.parallelism)
        self.right_lane.add_fit(right_located_line, self.parallelism)

        self.locator_with_prior_visualized = locatorWithPrior.visualize()

        # calculate distance between the two identified lane lines
        self.get_dist_between_lines()

    def visualize(self, panel_scale=0.3):
        """
        generate output image of undistorted original image and identified lane line,
        with a small panel displaying identified lane line on the top view perspective of undistorted original image
        :param panel_scale: scale of the panel on the output image
        :return: output image
        """
        thresholded = np.dstack((self.thresholded, self.thresholded, self.thresholded)) * 255

        panel_img = cv2.resize(thresholded, (0, 0), fx=panel_scale, fy=panel_scale)
        panel_img_2 = cv2.resize(self.locator_with_prior_visualized, (0, 0), fx=panel_scale, fy=panel_scale)

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

        text = 'frame : %d, left curv: %.2f, right curv: %.2f' \
               % (self.frame, self.left_lane.radius_of_curvature, self.right_lane.radius_of_curvature)
        cv2.putText(result, text, (10, result.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        text = 'center dist: %.2f meters, parallelism %.2f' % (self.line_base_pos * Lane.Y_M_PER_PIX, self.parallelism)
        cv2.putText(result, text, (10, result.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        overlay = cv2_overlay(result, panel_img, (5, 5))
        overlay = cv2_overlay(overlay, panel_img_2, (10 + int(self.width * panel_scale), 5))
        return overlay

    def get_dist_between_lines(self):
        """
        get the distance between the left and right lines
        :return: None
        """
        self.dist_between_lines.append(np.average(self.right_lane.average_fitx - self.left_lane.average_fitx))
        self.average_dist_between_lines = np.average(self.dist_between_lines)

    def center_offset(self):
        """
        calculate the distance in meters of vehicle center from the center of the lane lines
        :return: None
        """
        self.line_base_pos = (self.left_lane.average_fitx[-1] + self.right_lane.average_fitx[-1]) / 2 - self.width / 2

    def get_parallelism(self, left_line, right_line):
        """
        how parallel the identified lines are
        :return:
        """
        dist = left_line.fitx - right_line.fitx
        avg = np.average(dist)
        self.parallelism = math.exp(-pow(np.average((dist/avg - 1) ** 2), 0.5))
        if math.isnan(self.parallelism):
            self.parallelism = 0.0


class Line:
    """
    Line on a image, can be either the left or right lane line
    """
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

        # radius of curvature of the line in some units
        self.radius_of_curvature = None

    def add_fit(self, located_line, parallilsm):
        """
        add a new located_line to the located_line in previously identified lane lines
        :param located_line: line identified and fit
        :param parallilsm: how parallel this lane line to the other lane line identified on the same image
        :return: None
        """
        # how parallel the lane lines are is used as a measurment of confidence of the lane lines we identified
        confidence = parallilsm

        self.detected = True
        self.recent_fitx.append(located_line.fitx)

        # confidence also consider the current identified line are not too far from previous line
        confidence = min(confidence, self.change_from_previous_line())

        if len(self.recent_fitx) > self.N:
            self.recent_fitx.pop(0)
        self.update_average_fitx(confidence)

        self.fity = located_line.fity

        self.recent_fit_coefficients.append(located_line.fit_coefficients)
        self.update_average_fit(confidence)
        self.fit_on_average = np.polyfit(self.fity, self.average_fitx, 2)

        # To choose current fit (using one of the three methods below):
        # self.fit = self.recent_fit_coefficients[-1]
        # self.fit = self.average_fit_coefficients
        self.fit = self.fit_on_average

        x_in_meter = self.average_fitx * Lane.X_M_PER_PIX
        y_in_meter = self.fity * Lane.Y_M_PER_PIX
        fit_im_meter = np.polyfit(y_in_meter, x_in_meter, 2)

        # self.radius_of_curvature = self.curvature(self.fit, 0) # in pixels
        self.radius_of_curvature = self.curvature(fit_im_meter, 2)

        self.line_x = located_line.line_x
        self.line_y = located_line.line_y

    def update_average_fitx(self, confidence=None):
        """
        update average fitx
        :param confidence: weight of current fitx in average fitx
        :return: None
        """
        if confidence is None:
            confidence = 1 / self.N
        if self.average_fitx is None:
            self.average_fitx = self.recent_fitx[-1]
        else:
            self.average_fitx = self.average_fitx * (1 - confidence) + self.recent_fitx[-1] * confidence

    def update_average_fit(self, confidence=None):
        """
        update average fit coefficients
        :param confidence: weight of current fit coefficients in average fit coefficients
        :return: None
        """
        if confidence is None:
            confidence = 1 / self.N
        if self.average_fit_coefficients is None:
            self.average_fit_coefficients = self.recent_fit_coefficients[-1]
        else:
            self.average_fit_coefficients = self.average_fit_coefficients * (1 - confidence) \
                                            + self.recent_fit_coefficients[-1] * confidence

    def change_from_previous_line(self):
        """
        use "distance of current line from last identified line in pixels, divided by 100 pixels"
        as a measurement of confidence of currently identified line
        notice only lower half of the lines (in top view) are considered!
        :return: None
        """
        if len(self.recent_fitx) < 2:
            return 1
        else:
            diff_recent_fitx = self.recent_fitx[-1] - self.recent_fitx[-2]
            return 1 - np.average(diff_recent_fitx[len(diff_recent_fitx) // 2:]) / 100

    @staticmethod
    def curvature(polynormial_factors, y):
        """
        calculate curvature
        :param polynormial_factors: polynormial factors
        :param y: y
        :return: curavure in radius
        """
        return pow(1 + pow(2 * polynormial_factors[0] * y + polynormial_factors[1], 2), 1.5) / abs(2 * polynormial_factors[0])


def cv2_overlay(image, panel, offset):
    """
    overlay a panel to a image, given its offset, with 70% weight of the panel at its target position on the image
    :param image: image, with larger shape
    :param panel: panel, with smaller shape, overlay on a region of the image
    :param offset: x, y coordinates of the upper left corner of the panel on the image
    :return: the overlayed image
    """
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
    lane = Lane(threshold_pipeline)
    lane.find_line(image)
    result = lane.visualize()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    plt.imshow(result)
    output_path = cfg.join_path(cfg.line_finder['output'], image_name + '_line.jpg')
    plt.show()
    plt.savefig(output_path)


if __name__ == '__main__':
    # test()

    lane = Lane(threshold_pipeline)
    # lane = Lane(challenge_threshold_pipeline)

    def process_image(image):
        """
        process image (identify the lane line), return the processed image
        :param image: input image
        :return: the processed image
        """
        lane.find_line(image)
        return lane.visualize()

    from moviepy.editor import VideoFileClip

    video_name = 'project_video.mp4'
    # video_name = 'challenge_video.mp4'
    # video_name = 'harder_challenge_video.mp4'

    input_path = cfg.join_path(cfg.video_path['videos'], video_name)
    output_path = cfg.join_path(cfg.video_path['output_videos'], video_name)

    # To speed up the testing process, only process a subclip of the first 5 seconds
    # clip1 = VideoFileClip(input_path).subclip(0, 5)
    clip1 = VideoFileClip(input_path)
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(output_path, audio=False)
