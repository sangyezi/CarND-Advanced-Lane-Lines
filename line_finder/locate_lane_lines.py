import numpy as np
import matplotlib.pyplot as plt
import cv2
import config as cfg
from line_finder.transform_perspective import TransformPerspective
from camera_calibration.undistort_image import Undistort


class LocatedLine:
    """
    Class to contain located line information
    """
    def __init__(self, fit_coefficients, line_x, line_y, height):
        """
        constructor
        :param fit_coefficients: polynomial coefficients of the fit
        :param line_x: x values for detected line pixels
        :param line_y: y values for detected line pixels
        """
        self.fit_coefficients = fit_coefficients
        self.line_x = line_x
        self.line_y = line_y

        self.fity = np.linspace(0, height - 1, height)
        self.fitx = Locator.polynormial_eval(fit_coefficients, self.fity)


class Locator:
    """
    The class to locate lane line on a thresholded and warped image using sliding window
    """
    def __init__(self, input_img, nwindows=9, margin=100, minpix=50):
        self.input_img = input_img
        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.input_img.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])

        # Create empty lists to receive left and right lane pixel indices
        self.left_lane_inds = []
        self.right_lane_inds = []

        self.left_windows = []
        self.right_windows = []

        self.left_fit = None
        self.right_fit = None

    @staticmethod
    def polynormial_eval(polynormial_factors, y):
        return polynormial_factors[0] * (y ** 2) + polynormial_factors[1] * y + polynormial_factors[2]

    @staticmethod
    def pixels_on_fit(height, left_fit, right_fit):
        ploty = np.linspace(0, height - 1, height)
        left_fitx = Locator.polynormial_eval(left_fit, ploty)
        right_fitx = Locator.polynormial_eval(right_fit, ploty)
        return ploty, left_fitx, right_fitx

    @staticmethod
    def cv2_draw_line(img, x, y, line_width, color):
        """
        recast the x and y points into usable format for cv2.fillPoly()
        :param img: img to draw on
        :param x: x coordinates of line
        :param y: y coordinates of line
        :param line_width: line widht in pixels
        :param color: line color
        :return: img with the line drawn on
        """
        line_left = np.array([np.transpose(np.vstack([x - line_width, y]))])
        line_right = np.array([np.flipud(np.transpose(np.vstack([x + line_width, y])))])
        line_pts = np.hstack((line_left, line_right))
        cv2.fillPoly(img, np.int_([line_pts]), color)
        return img

    def sliding_window(self):
        """
        using sliding window to find fit onf the lane line
        :param self.input_img: use a thresholded and warped image
        :param nwindows: the number of sliding windows
        :param margin: the width of the windows +/- margin
        :param minpix: the minimum number of pixels found to recenter window
        :return: left_fit, right_fit: quadratic fit of the lane line
        """
        # Take a histogram of the bottom half of the image
        histogram = np.sum(self.input_img[self.input_img.shape[0] // 2:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(self.input_img.shape[0] / self.nwindows)

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.input_img.shape[0] - (window + 1) * window_height
            win_y_high = self.input_img.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            self.left_windows.append([(win_xleft_low, win_y_low), (win_xleft_high, win_y_high)])
            self.right_windows.append([(win_xright_low, win_y_low), (win_xright_high, win_y_high)])

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high)
                              & (self.nonzerox >= win_xleft_low) & (self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high)
                               & (self.nonzerox >= win_xright_low) & (self.nonzerox < win_xright_high)).nonzero()[0]
            # Extend the list of good indices
            self.left_lane_inds.extend(good_left_inds)
            self.right_lane_inds.extend(good_right_inds)

            # If found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))

        # Extract left and right line pixel positions
        leftx = self.nonzerox[self.left_lane_inds]
        lefty = self.nonzeroy[self.left_lane_inds]
        rightx = self.nonzerox[self.right_lane_inds]
        righty = self.nonzeroy[self.right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        return LocatedLine(self.left_fit, leftx, lefty, self.input_img.shape[0]), \
               LocatedLine(self.right_fit, rightx, righty, self.input_img.shape[0])

    def visualize(self):
        """
        visualize lane line on the warped image
        :return: np array representing the visualization
        """
        if self.left_fit is not None:
            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((self.input_img, self.input_img, self.input_img))
            for left_window in self.left_windows:
                cv2.rectangle(out_img, left_window[0], left_window[1], (0, 255, 0), 2)

            for right_window in self.right_windows:
                cv2.rectangle(out_img, right_window[0], right_window[1], (0, 255, 0), 2)

            ploty, left_fitx, right_fitx = Locator.pixels_on_fit(self.input_img.shape[0], self.left_fit, self.right_fit)

            out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
            out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]

            self.cv2_draw_line(out_img, left_fitx, ploty, 3, (255, 255, 0))
            self.cv2_draw_line(out_img, right_fitx, ploty, 3, (255, 255, 0))

            return out_img


class LocatorWithPrior(Locator):
    """
    visualize lane line on the warped image
    """
    def __init__(self, input_img, prior_left_fit, prior_right_fit, nwindows=9, margin=100, minpix=50):
        super().__init__(input_img, nwindows, margin, minpix)
        self.prior_left_fit = prior_left_fit
        self.prior_right_fit = prior_right_fit

    def sliding_window(self):
        self.left_lane_inds = ((self.nonzerox > self.polynormial_eval(self.prior_left_fit, self.nonzeroy) - self.margin)
                               & (self.nonzerox < self.polynormial_eval(self.prior_left_fit, self.nonzeroy) + self.margin))
        self.right_lane_inds = ((self.nonzerox > self.polynormial_eval(self.prior_right_fit, self.nonzeroy) - self.margin)
                                & (self.nonzerox < self.polynormial_eval(self.prior_right_fit, self.nonzeroy) + self.margin))

        # Again, extract left and right line pixel positions
        leftx = self.nonzerox[self.left_lane_inds]
        lefty = self.nonzeroy[self.left_lane_inds]
        rightx = self.nonzerox[self.right_lane_inds]
        righty = self.nonzeroy[self.right_lane_inds]
        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        return LocatedLine(self.left_fit, leftx, lefty, self.input_img.shape[0]), \
               LocatedLine(self.right_fit, rightx, righty, self.input_img.shape[0])
        
    def visualize(self):
        """
        visualize lane line on the warped image
        :return: np array representing the visualization
        """
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((self.input_img, self.input_img, self.input_img))
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]

        # Generate x and y values for plotting
        ploty, left_fitx, right_fitx = Locator.pixels_on_fit(self.input_img.shape[0], self.left_fit, self.right_fit)

        # Generate a polygon to illustrate the search window area
        # Draw the lane onto the warped blank image
        self.cv2_draw_line(window_img, left_fitx, ploty, self.margin, (0, 255, 0))
        self.cv2_draw_line(window_img, right_fitx, ploty, self.margin, (0, 255, 0))

        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # result[ploty.astype(int), left_fitx.astype(int), :] = [255, 255, 0]
        # result[ploty.astype(int), right_fitx.astype(int), :] = [255, 255, 0]

        self.cv2_draw_line(result, left_fitx, ploty, 3, (255, 255, 0))
        self.cv2_draw_line(result, right_fitx, ploty, 3, (255, 255, 0))

        return result
        # plt.imshow(result)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')


def main():
    image_name = 'test6'
    threshold_image_name = image_name + '_threshold'
    input_path = cfg.join_path(cfg.line_finder['output'], threshold_image_name + '.jpg')
    input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    transformer = TransformPerspective()

    input_img = transformer.transform(input_img)

    locator = Locator(input_img)
    left_located_line, right_located_line = locator.sliding_window()
    left_fit = left_located_line.fit_coefficients
    right_fit = right_located_line.fit_coefficients
    out_img = locator.visualize()
    plt.imshow(out_img)
    output_path = cfg.join_path(cfg.line_finder['output'], threshold_image_name + '_line1.jpg')
    plt.savefig(output_path)

    locatorWithPrior = LocatorWithPrior(input_img, left_fit, right_fit)
    left_located_line, right_located_line=locatorWithPrior.sliding_window()
    left_fit = left_located_line.fit_coefficients
    right_fit = right_located_line.fit_coefficients
    out_img = locatorWithPrior.visualize()
    plt.imshow(out_img)
    output_path = cfg.join_path(cfg.line_finder['output'], threshold_image_name + '_line2.jpg')
    plt.savefig(output_path)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(input_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty, left_fitx, right_fitx = Locator.pixels_on_fit(input_img.shape[0], left_fit, right_fit)

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = transformer.inverse_transform(color_warp)

    # Combine the result with the original image
    origin_path = cfg.join_path(cfg.line_finder['input'], image_name + '.jpg')
    origin_img = cv2.imread(origin_path)
    origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)

    undistort = Undistort()
    undist = undistort.undistort_image(origin_img)
    output_path = cfg.join_path(cfg.line_finder['output'], image_name + '_line.jpg')
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    plt.imshow(result)
    plt.savefig(output_path)

if __name__ == '__main__':
    main()
