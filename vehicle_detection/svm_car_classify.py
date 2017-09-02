import matplotlib.image as mpimg
import numpy as np
import glob
import time
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle
import config as cfg
from vehicle_detection.image_features import get_hog_features, convert_color, bin_spatial, color_hist, normalize_img


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_size=None, hist_bins=None):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # image = normalize_img(image)

        # apply color conversion if other than 'RGB'
        feature_image = convert_color(image, cspace)
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        if spatial_size:
            spatial_features = bin_spatial(image, size=spatial_size)
            hog_features = np.hstack((hog_features, spatial_features))
        if hist_bins:
            hist_features = color_hist(image, nbins=hist_bins)
            hog_features = np.hstack((hog_features, hist_features))

        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features


vehicle_path = cfg.vehicle_detection['vehicles']
cars = glob.glob(cfg.join_path(vehicle_path, '**/*.png'))
non_vehicle_path = cfg.vehicle_detection['non-vehicles']
notcars = glob.glob(cfg.join_path(non_vehicle_path, '**/*.png'))


# Reduce the sample size because HOG features are slow to compute
# The quiz evaluator times out after 13s of CPU time
# sample_size = 500
# cars = cars[0: sample_size]
# notcars = notcars[0: sample_size]

t = time.time()
car_features = extract_features(cars, cspace=cfg.vehicle_detection['color-space'],
                                orient=cfg.vehicle_detection['orient'],
                                pix_per_cell=cfg.vehicle_detection['pix_per_cell'],
                                cell_per_block=cfg.vehicle_detection['cell_per_block'],
                                hog_channel=cfg.vehicle_detection['hog_channel'],
                                spatial_size=cfg.vehicle_detection['spatial_size'],
                                hist_bins=cfg.vehicle_detection['hist_bins'])

notcar_features = extract_features(notcars, cspace=cfg.vehicle_detection['color-space'],
                                   orient=cfg.vehicle_detection['orient'],
                                   pix_per_cell=cfg.vehicle_detection['pix_per_cell'],
                                   cell_per_block=cfg.vehicle_detection['cell_per_block'],
                                   hog_channel=cfg.vehicle_detection['hog_channel'],
                                   spatial_size=cfg.vehicle_detection['spatial_size'],
                                   hist_bins=cfg.vehicle_detection['hist_bins'])

t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', cfg.vehicle_detection['orient'], 'orientations', cfg.vehicle_detection['pix_per_cell'],
      'pixels per cell and', cfg.vehicle_detection['cell_per_block'], 'cells per block')
print('Feature vector length:', len(X_train[0]))

t = time.time()

# Use a linear SVC
svc = svm.LinearSVC()

# parameters = {'kernel': ('linear', 'rbf'), 'C': [0.1, 1, 10]}
# svc = svm.SVC(kernel='rbf', C=10)
# svc = svm.SVC()
# clf = GridSearchCV(svc, parameters)

# clf.fit(X_train, y_train)
svc.fit(X_train, y_train)

# print(clf.best_params_)

t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

# Check the score of the SVC
# print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# Check the prediction time for a single sample
t = time.time()
n_predict = 10
# print('My SVC predicts:     ', clf.predict(X_test[0: n_predict]))
print('My SVC predicts:     ', svc.predict(X_test[0: n_predict]))

print('For these', n_predict, 'labels: ', y_test[0: n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict, ' labels with SVC')

dist_pickle = dict()
# dist_pickle["svc"] = clf
dist_pickle["svc"] = svc
dist_pickle["scaler"] = X_scaler
pickle.dump(dist_pickle, open(cfg.vehicle_detection['svc_file'], "wb"))
