import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from vehicle_detection.image_features import convert_color, get_hog_features, bin_spatial, color_hist, normalize_img
import config as cfg


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, cspace, hog_channel, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    rectangles = []

    img = normalize_img(img)

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, cspace)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    # Define blocks and steps as above
    nxblocks = (ctrans_tosearch.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ctrans_tosearch.shape[0] // pix_per_cell) - cell_per_block + 1
    # nfeat_per_block = orient * cell_per_block ** 2  # number of features per block

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # use feature_vec=False to not ravel, so can extract each window
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    else:
        ch1 = ctrans_tosearch[:, :, hog_channel]
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch

            if hog_channel == 'ALL':
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            features = hog_features

            # Get color features
            if spatial_size:
                spatial_features = bin_spatial(subimg, size=spatial_size)
                features = np.hstack((features, spatial_features))

            if hist_bins:
                hist_features = color_hist(subimg, nbins=hist_bins)
                features = np.hstack((features, hist_features))

            # Scale features and make a prediction
            test_features = X_scaler.transform(features.reshape(1, -1))

            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                rectangles.append(((xbox_left, ytop_draw + ystart), (xbox_left+win_draw, ytop_draw + win_draw+ystart)))

    return rectangles

# from trained model
dist_pickle = pickle.load(open(cfg.vehicle_detection['svc_file'], "rb"))
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]

# config values
cspace = cfg.vehicle_detection['color-space']
orient = cfg.vehicle_detection['orient']
pix_per_cell = cfg.vehicle_detection['pix_per_cell']
cell_per_block = cfg.vehicle_detection['cell_per_block']
hog_channel = cfg.vehicle_detection['hog_channel']
spatial_size = cfg.vehicle_detection["spatial_size"]
hist_bins = cfg.vehicle_detection["hist_bins"]


image_name = 'test6'
image_path = cfg.join_path(cfg.line_finder['input'], image_name + '.jpg')
img = mpimg.imread(image_path)
img_scaled = img.astype(np.float32) / 255

# specific to car image problem
ystart = 400
ystop = 656
scale = 1.5

rectangles = find_cars(img_scaled, ystart, ystop, scale, svc, X_scaler, cspace, hog_channel, orient, pix_per_cell,
                       cell_per_block, spatial_size, hist_bins)

for rectangle in rectangles:
    cv2.rectangle(img, rectangle[0], rectangle[1], (0, 0, 255), 6)
plt.imshow(img)
plt.show()
