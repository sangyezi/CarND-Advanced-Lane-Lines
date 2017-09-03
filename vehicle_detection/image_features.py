import numpy as np
import cv2
from skimage.feature import hog


def convert_color(image, cspace):
    if cspace != 'RGB':
        if cspace == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        return np.copy(image)


def normalize_img(img):
    img = img.astype(np.float32)
    max = np.max(img.ravel())
    return img / max


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def main():
    import config as cfg
    import glob
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    vehicle_path = cfg.vehicle_detection['vehicles']
    cars = glob.glob(cfg.join_path(vehicle_path, '**/*.png'))
    non_vehicle_path = cfg.vehicle_detection['non-vehicles']
    notcars = glob.glob(cfg.join_path(non_vehicle_path, '**/*.png'))

    car_image = mpimg.imread(cars[np.random.randint(0, len(cars))])
    notcar_image = mpimg.imread(notcars[np.random.randint(0, len(notcars))])

    orient = cfg.vehicle_detection['orient']
    pix_per_cell = cfg.vehicle_detection['pix_per_cell']
    cell_per_block = cfg.vehicle_detection['cell_per_block']

    features, car_hog = get_hog_features(car_image[:, :, 2], orient, pix_per_cell, cell_per_block,
                                         vis=True, feature_vec=True)
    features, notcar_hog = get_hog_features(notcar_image[:, :, 2], orient, pix_per_cell, cell_per_block,
                                            vis=True, feature_vec=True)

    fig = plt.figure(figsize=(12, 10))
    plt.subplot(221)
    plt.imshow(car_image)
    plt.title('Car', fontsize=18)
    plt.subplot(222)
    plt.imshow(notcar_image)
    plt.title('Not Car', fontsize=18)
    plt.subplot(223)
    plt.imshow(car_hog, cmap='gray')
    plt.title('Car HOG', fontsize=18)
    plt.subplot(224)
    plt.imshow(notcar_hog, cmap='gray')
    plt.title('Not Car HOG', fontsize=18)

    fig.tight_layout()
    # plt.show()
    plt.savefig(cfg.join_path(cfg.vehicle_detection['output'], 'hog.jpg'))

if __name__ == '__main__':
    main()
