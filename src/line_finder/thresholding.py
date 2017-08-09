import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def get_sobelx(img, ksize=5):
    """
    calculate sobel derivative in x direction, take absolute value and scale it
    :param img: image as input
    :return: sobel x (absolute value, scaled) of the image
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    return scaled_sobelx


def get_sobely(img, ksize=5):
    """
    calculate sobel derivative in y direction, take absolute value and scale it
    :param img: image as input
    :return: sobel y(absolute value, scaled) of the image
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    return scaled_sobelx


def get_sobel_mag(img, ksize=3):
    """
    get sobel magnitude
    :param img: image as input
    :param ksize: sobel magnitude
    :return: sobel direction
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize)

    sobel_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    sobel_mag = (sobel_mag * 255 / np.max(sobel_mag)).astype(np.uint8)
    return sobel_mag


def get_sobel_direct(img, ksize=3):
    """
    get sobel direction
    :param img: image as input
    :param ksize: sobel direction
    :return:
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize)

    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)

    grad_direct = np.arctan2(abs_sobely, abs_sobelx)

    return grad_direct


def threshold_binary(img, thresh):
    """
    convert single color image (pixel value: 0-255) to binary image (pixel value: 0-1)
    :param img: single color image
    :param thresh: within thresh, convert to 1, outside of thresh, convert to 0
    :return: converted image
    """
    if len(img.shape) == 2 or img.shape[2] == 1:
        binary = np.zeros_like(img)
        binary[(img >= thresh[0]) & (img < thresh[1])] = 1
        return binary


def combine_binary_or(img, *images):
    """
    combine 1 or more binary images using OR
    :param img: the first binary image
    :param images: the rest of the binary images
    :return: the combined binary image
    """
    combined = np.zeros_like(img)
    for image in images:
        img = (img == 1) | (image == 1)
    combined[img == 1] = 1
    return combined


def combine_binary_and(img, *images):
    """
    combine 1 or more binary images using AND
    :param img: the first binary image
    :param images: the rest of the binary images
    :return: the combined binary image
    """
    combined = np.zeros_like(img)
    for image in images:
        img = (img == 1) & (image == 1)
    combined[img == 1] = 1
    return combined


def color_stack(*images):
    """
    stack 2 or 3 images in rgb space, note if original image is binary, consider scale after stack
    :param images: images to stack
    :return: stacked image
    """
    if len(images) == 2:
        return np.dstack((np.zeros_like(images[0]), images[0], images[1]))
    if len(images) == 3:
        return np.dstack((images[0], images[1], images[2]))


def image_threshhold_filter(img, color_thresh, sobel_mag_thresh, sobel_direct_thresh):
    """
    generate filtered images
    :param img: original image (should use undistorted 3-channel image)
    :param color_thresh: threshold for color
    :param sobel_mag_thresh: treshold for sobel magnitude
    :param sobel_direct_thresh: threshold for sobel direction
    :return: binary images represent color filtered, sobel magnitude filtered and sobel direction filtered

    Note: gray or red channel are not as good as s channel, in any regards
    also, sobel_mag and sobel_direction are more general than sobelx and sobely
    So those processing were not used in final result
    """
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # r_channel = img[:, :, 2]
    s_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]

    # gray_binary = threshold_binary(gray, color_thresh)
    # r_binary = threshold_binary(r_channel, color_thresh)
    s_binary = threshold_binary(s_channel, color_thresh)

    color_binary = s_binary

    single_channel = s_channel
    # sobelx = get_sobelx(single_channel)
    # sobelx_binary = threshold_binary(sobelx, sobel_mag_thresh)

    # sobely = get_sobely(single_channel)
    # sobely_binary = threshold_binary(sobely, sobel_mag_thresh)

    sobel_mag = get_sobel_mag(single_channel)
    sobel_mag_binary = threshold_binary(sobel_mag, sobel_mag_thresh)

    sobel_direct = get_sobel_direct(single_channel)
    sobel_direct_binary = threshold_binary(sobel_direct, sobel_direct_thresh)

    return color_binary, sobel_mag_binary, sobel_direct_binary


def plot(img1, img2, img3, color_binary, combined_binary, combined_image_masked, text):
    """
    plot images
    :param img1: image 1
    :param img2: image 2
    :param img3: image 3
    :param color_binary: image 4
    :param combined_binary:  image 5
    :param combined_image_masked: image 6
    :param text: figure title
    :return: None
    """
    f, axes = plt.subplots(2, 3, figsize=(20, 10))
    f.suptitle(text)

    axes[0, 0].set_title('color channel')
    axes[0, 0].imshow(img1, cmap='gray')

    axes[0, 1].set_title('sobel mag')
    axes[0, 1].imshow(img2, cmap='gray')

    axes[0, 2].set_title('sobel direct')
    axes[0, 2].imshow(img3, cmap='gray')

    axes[1, 0].set_title('r: color, g: sobel mag, b: sobel direct')
    axes[1, 0].imshow(color_binary * 128)

    axes[1, 1].set_title('Combined')
    axes[1, 1].imshow(combined_binary, cmap='gray')

    axes[1, 2].set_title('Combined and masked')
    axes[1, 2].imshow(combined_image_masked, cmap='gray')

    plt.show()


def generate_roi(image, apex_relative_pos, apex_relative_width):
    """
    generate region of interest (polygon with four vertices)
    :param image: input image
    :param apex_relative_pos: relative position of apex
    :param apex_relative_width:  relative width of apex
    :return: four vertices of the region of interest)
    """
    height = image.shape[0]
    width = image.shape[1]

    apex = (apex_relative_pos[0] * width, apex_relative_pos[1] * height)
    apex_width = apex_relative_width * width

    vertices = np.array([[(0, height),(apex[0] - apex_width, apex[1]),
                          (apex[0] + apex_width, apex[1]), (width, height)]], dtype=np.int32)

    return vertices


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def pipeline(img, to_plot=False):
    """
    the tested pipeline for thresholding
    :param img: image as input
    :param to_plot: if True, plot and show the processed image
    :return: binary image where 1 represent lane line, 0 represent not lane line
    """

    color_thresh = (200, 255)
    sobel_mag_thresh = (40, 255)
    sobel_direct_thresh = (0.7, 1.3)
    apex_relative_pos = (0.5, 0.6)
    apex_relative_width = 0.05

    color_binary, sobel_mag_binary, sobel_direct_binary = image_threshhold_filter(img, color_thresh=color_thresh,
                                                                                  sobel_mag_thresh=sobel_mag_thresh,
                                                                                  sobel_direct_thresh=sobel_direct_thresh)

    color_binary_stacked = color_stack(color_binary, sobel_mag_binary, sobel_direct_binary)

    # final output
    combined_binary = combine_binary_or(color_binary, combine_binary_and(sobel_mag_binary, sobel_direct_binary))
    combined_binary_masked = region_of_interest(combined_binary, generate_roi(combined_binary,
                                                                              apex_relative_pos=apex_relative_pos,
                                                                              apex_relative_width=apex_relative_width))

    if to_plot:
        fig_title = "%s color_thresh: %s, sobel_mag_thresh: %s, sobel_direct_thresh %s" \
                    % (image_name, color_thresh, sobel_mag_thresh, sobel_direct_thresh)

        plot(color_binary, sobel_mag_binary, combine_binary_and(sobel_mag_binary, sobel_direct_binary),
             color_binary_stacked, combined_binary, combined_binary_masked, fig_title)

    return combined_binary_masked


if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    image_name = 'test3'
    image_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'test_images', image_name + '.jpg'))
    img = cv2.imread(image_path)
    line_binary = pipeline(img, True)

    line = np.dstack((line_binary * 255, np.zeros_like(line_binary), np.zeros_like(line_binary)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_line = cv2.addWeighted(img, 1, line, 1, 0)
    #todo: plot img_line