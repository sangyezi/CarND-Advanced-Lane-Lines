from camera_calibration.undistort_image import Undistort
from line_finder.transform_perspective import TransformPerspective

from line_finder.thresholding import threshold_pipeline
from line_finder.locate_lane_lines import Locator, LocatorWithPrior

import cv2
import numpy as np
import config as cfg
import matplotlib.pyplot as plt
from line_finder.workflow import cv2_overlay


# prototype of the workflow, actually replaced by workflow.py
def pipeline_prototype():
    panel_scale = 0.33
    image_name = 'test6'
    image_path = cfg.join_path(cfg.line_finder['input'], image_name + '.jpg')
    img = cv2.imread(image_path)

    undistort = Undistort()
    undist = undistort.undistort_image(img)
    combined_binary, combined_binary_masked = threshold_pipeline(undist, False)

    input_img = combined_binary

    transformer = TransformPerspective()
    input_img = transformer.transform(input_img)

    locator = Locator(input_img)
    left_located_line, right_located_line = locator.sliding_window()
    left_fit = left_located_line.fit_coefficients
    right_fit = right_located_line.fit_coefficients
    out_img_1 = cv2.resize(locator.visualize(), (0, 0), fx=panel_scale, fy=panel_scale)

    locatorWithPrior = LocatorWithPrior(input_img, left_fit, right_fit)
    left_located_line, right_located_line = locatorWithPrior.sliding_window()
    left_fit = left_located_line.fit_coefficients
    right_fit = right_located_line.fit_coefficients
    out_img_2 = cv2.resize(locatorWithPrior.visualize(), (0, 0), fx=panel_scale, fy=panel_scale)

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

    undist = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    result = cv2_overlay(result, out_img_1, (5, 5))
    result = cv2_overlay(result, out_img_2, (5 + out_img_1.shape[1], 5))

    plt.imshow(result)
    output_path = cfg.join_path(cfg.line_finder['output'], image_name + '_line.jpg')
    plt.show()
    # plt.savefig(output_path)

if __name__ == '__main__':
    pipeline_prototype()

