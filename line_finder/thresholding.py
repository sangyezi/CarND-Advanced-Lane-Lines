import cv2
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from camera_calibration.undistort_image import Undistort


class Channel:
    """
    class to extract a single channel from a image (BGR channels)
    """
    @staticmethod
    def gray(img, clahe=False):
        if clahe:
            img = Channel.clahe_bgr(img)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def red_channel(img, clahe=False):
        if clahe:
            img = Channel.clahe_bgr(img)
        return img[:, :, 2]

    @staticmethod
    def green_channel(img, clahe=False):
        if clahe:
            img = Channel.clahe_bgr(img)
        return img[:, :, 1]

    @staticmethod
    def blue_channel(img, clahe=False):
        if clahe:
            img = Channel.clahe_bgr(img)
        return img[:, :, 0]

    @staticmethod
    def h_channel(img, clahe=False):
        if clahe:
            img = Channel.clahe_bgr(img)
        return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 0]

    @staticmethod
    def l_channel(img, clahe=False):
        if clahe:
            img = Channel.clahe_bgr(img)
        return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 1]

    @staticmethod
    def s_channel(img, clahe=False):
        if clahe:
            img = Channel.clahe_bgr(img)
        return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]

    @staticmethod
    def clahe_bgr(img):
        """
        Histogram Equalization
        :param img: image with 3 channels (B, G, R)
        :return: equalized image
        """
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


class Gradient:
    """
    get gradient on a single channel image
    """
    @staticmethod
    def sobel_x(img, ksize=3):
        """
        calculate sobel derivative in x direction, take absolute value and scale it
        :param img: image as input
        :return: sobel x (absolute value, scaled) of the image
        """
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        return scaled_sobelx

    @staticmethod
    def sobel_y(img, ksize=3):
        """
        calculate sobel derivative in y direction, take absolute value and scale it
        :param img: image as input
        :return: sobel y(absolute value, scaled) of the image
        """
        sobelx = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        return scaled_sobelx

    @staticmethod
    def sobel_magnitude(img, ksize=3):
        """
        get sobel magnitude
        :param img: image as input
        :param ksize: sobel magnitude
        :return: sobel magnitude
        """
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize)

        sobel_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

        sobel_mag = (sobel_mag * 255 / np.max(sobel_mag)).astype(np.uint8)
        return sobel_mag

    @staticmethod
    def sobel_direction(img, ksize=3):
        """
        get sobel direction
        :param img: image as input
        :param ksize: sobel direction
        :return: sobel direction
        """
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize)

        abs_sobelx = np.abs(sobelx)
        abs_sobely = np.abs(sobely)

        grad_direct = np.arctan2(abs_sobely, abs_sobelx)

        return grad_direct


class Thresholding:
    """
    generate binary image by thresholding single channel images or combining thresholded binary images
    """
    @staticmethod
    def range(img, thresh):
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


class Module:
    @staticmethod
    def transform(img, channel_functor, gradient_functor, clahe=False):
        """
        transform a image by (optionally) equalize exposure, (always) getting a single channel
        and (optionally) apply gradient
        :param img: input image, should have BGR channels, pixels value in (0, 255)
        :param channel_functor: to get a single channel
        :param gradient_functor:  to get gradient (use None if no use gradient)
        :param clahe: to equalize channel
        :return:
        """
        if gradient_functor:
            functor = lambda img: gradient_functor(channel_functor(img, clahe))
            processed = functor(img)
            name = "%s_%s" % (channel_functor.__name__, gradient_functor.__name__)
        else:
            functor = channel_functor
            processed = functor(img)
            name = "%s" % channel_functor.__name__
        return name, processed

    @staticmethod
    def filter(name, img, thresh_functor, thresh):
        """
        filter a transformed image by a thresh, pixels within thresh get value 1, outside thresh get value 0
        resulted in a binary image
        :param named: string as img name
        :param img: input image, should have a single channel, pixels value in (0, 255)
        :param thresh_functor: functor to filter image
        :param thresh: to filter image
        :return:
        """
        name = "_%s_%s" % (name, thresh_functor.__name__)
        processed = thresh_functor(img, thresh)
        return name, processed


    @staticmethod
    def module(img, channel_functor, gradient_functor, thresh_functor, thresh, clahe=False):
        """
        combination of transform and filter
        :param img:
        :param channel_functor:
        :param gradient_functor:
        :param thresh_functor:
        :param thresh:
        :param clahe:
        :return:
        """
        name, processed = Module.transform(img, channel_functor, gradient_functor, clahe)
        name, processed = Module.filter(name, processed, thresh_functor, thresh)
        return name, processed


class Combination:
    @staticmethod
    def any(img, *images):
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

    @staticmethod
    def all(img, *images):
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

    @staticmethod
    def color_stack(*images):
        """
        stack 2 or 3 images in rgb space, note if original image is binary, consider scale after stack
        :param images: images to stack
        :return: stacked image (3 channels)
        """
        if len(images) == 2:
            return np.dstack((np.zeros_like(images[0]), images[0], images[1]))
        if len(images) == 3:
            return np.dstack((images[0], images[1], images[2]))


class RegionOfInterest:
    """
    generate region of interest in a image
    """
    @staticmethod
    def get_vertices(image, apex_relative_pos, apex_relative_width):
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

    @staticmethod
    def roi(img, vertices):
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


class ImageList:
    """
    util class to deal with a list of images
    """
    def __init__(self, images):
        self.images = images

    def get(self, title):
        for image_title, image in self.images:
            if image_title == title:
                return image

    def plot(self):
        import math
        size = len(self.images)
        m = int(math.ceil(math.sqrt(size)))
        n = int(math.ceil(size / m))

        f, axes = plt.subplots(n, m, figsize=(m * 5, n * 5))
        for i in range(0, n):
            for j in range(0, m):
                if i * m + j >= size:
                    break
                title, image = self.images[i * m + j]
                axes[i, j].set_title(title)
                if len(image.shape) == 3 and image.shape[2] == 3:
                    axes[i, j].imshow(image)
                else:
                    axes[i, j].imshow(image, cmap='gray')

        plt.show()


def threshold_pipeline(img):
    """
    thresholding pipeline, boilerplate code in the workflow
    :param img: image as input
    :return: ImageList: a list of images (as intermediate and ultimate processed result)
    """
    modules = [Module.module(img, Channel.gray, None, Thresholding.range, (130, 255), False),
               Module.module(img, Channel.s_channel, None, Thresholding.range, (100, 255), False),
               Module.module(img, Channel.gray, Gradient.sobel_x, Thresholding.range, (20, 100), False),
               Module.module(img, Channel.gray, Gradient.sobel_direction, Thresholding.range, (0.5, 1.2), False)]

    thresholded = Combination.any(
        Combination.all(
            modules[0][1],
            modules[1][1]
        ),
        Combination.all(
            modules[2][1],
            modules[3][1]
        )
    )

    stacked = Combination.color_stack(Combination.all(modules[0][1], modules[1][1]),
                                      modules[2][1], modules[3][1])

    apex_relative_pos = (0.5, 0.6)
    apex_relative_width = 0.05
    thresholded_masked = RegionOfInterest.roi(thresholded, RegionOfInterest.get_vertices(thresholded,
                                                                            apex_relative_pos=apex_relative_pos,
                                                                            apex_relative_width=apex_relative_width))

    modules.append(('thresholded', thresholded))
    modules.append(('stacked', stacked))
    modules.append(('thresholded_masked', thresholded_masked))
    modules.append(('original', cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return ImageList(modules)


def challenge_threshold_pipeline(img):
    """
    thresholding pipeline, boilerplate code in the workflow
    :param img: image as input
    :return: ImageList: a list of images (as intermediate and ultimate processed result)
    """
    modules = [Module.module(img, Channel.s_channel, None, Thresholding.range, (80, 255), False),
               Module.module(img, Channel.blue_channel, None, Thresholding.range, (180, 255), False),
               Module.module(img, Channel.gray, Gradient.sobel_x, Thresholding.range, (15, 100), False),
               Module.module(img, Channel.gray, Gradient.sobel_direction, Thresholding.range, (0.5, 1.2), False)]
    thresholded = Combination.all(
        Combination.any(
            modules[0][1],
            modules[1][1],
        ),
        Combination.all(
            modules[2][1],
            modules[3][1]
        )
    )

    stacked = Combination.color_stack(Combination.all(modules[0][1], modules[1][1]),
                                      modules[2][1], modules[3][1])

    apex_relative_pos = (0.5, 0.6)
    apex_relative_width = 0.05
    thresholded_masked = RegionOfInterest.roi(thresholded, RegionOfInterest.get_vertices(thresholded,
                                                                            apex_relative_pos=apex_relative_pos,
                                                                            apex_relative_width=apex_relative_width))

    modules.append(('thresholded', thresholded))
    modules.append(('stacked', stacked))
    modules.append(('thresholded_masked', thresholded_masked))
    modules.append(('original', cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return ImageList(modules)


def challenge_threshold_pipeline_2(img):
    """
    thresholding pipeline, boilerplate code in the workflow
    :param img: image as input
    :return: ImageList: a list of images (as intermediate and ultimate processed result)
    """
    modules = [Module.module(img, Channel.l_channel, None, Thresholding.range, (175, 255), False),
               Module.module(img, Channel.s_channel, None, Thresholding.range, (77, 255), False)]
    thresholded = Combination.any(
        modules[0][1],
        modules[1][1]
    )

    stacked = Combination.color_stack(modules[0][1], modules[1][1])

    apex_relative_pos = (0.5, 0.6)
    apex_relative_width = 0.05
    thresholded_masked = RegionOfInterest.roi(thresholded, RegionOfInterest.get_vertices(thresholded,
                                                                            apex_relative_pos=apex_relative_pos,
                                                                            apex_relative_width=apex_relative_width))

    modules.append(('thresholded', thresholded))
    modules.append(('stacked', stacked))
    modules.append(('thresholded_masked', thresholded_masked))
    modules.append(('original', cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return ImageList(modules)


def test_threshold_pipeline():
    image_name = 'test5'

    image_path = cfg.join_path(cfg.line_finder['input'], image_name + '.jpg')
    img = cv2.imread(image_path)

    undistort = Undistort()
    img = undistort.undistort_image(img)
    undist_path = cfg.join_path(cfg.line_finder['output'], image_name + '_undist.jpg')
    cv2.imwrite(undist_path, img)

    images = threshold_pipeline(img)
    images.plot()

    thresholded_path = cfg.join_path(cfg.line_finder['output'], image_name + '_threshold.jpg')
    thresholded_mask_path = cfg.join_path(cfg.line_finder['output'], image_name + '_threshold_masked.jpg')
    img_line_path = cfg.join_path(cfg.line_finder['output'], image_name + '_threshold_lane.jpg')

    thresholded_binary = images.get('thresholded')
    thresholded_masked = images.get('thresholded_masked')

    line = np.dstack((thresholded_masked * 255, np.zeros_like(thresholded_masked), np.zeros_like(thresholded_masked)))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_line = cv2.addWeighted(img, 1, line, 1, 0)
    img_line = cv2.cvtColor(img_line, cv2.COLOR_BGR2RGB)

    cv2.imwrite(thresholded_path, thresholded_binary * 255)
    cv2.imwrite(thresholded_mask_path, thresholded_masked * 255)
    cv2.imwrite(img_line_path, img_line)

if __name__ == '__main__':
    test_threshold_pipeline()

