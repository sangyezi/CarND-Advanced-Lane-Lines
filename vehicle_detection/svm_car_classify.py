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
def extract_features(imgs):
    cspace = cfg.vehicle_detection['color-space']
    orient = cfg.vehicle_detection['orient']
    pix_per_cell = cfg.vehicle_detection['pix_per_cell']
    cell_per_block = cfg.vehicle_detection['cell_per_block']
    hog_channel = cfg.vehicle_detection['hog_channel']
    spatial_size = cfg.vehicle_detection['spatial_size']
    hist_bins = cfg.vehicle_detection['hist_bins']

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


test_size = 0.2

vehicle_path = cfg.vehicle_detection['vehicles']

cars_GTI_folders = ['GTI_Far', 'GTI_Left', 'GTI_MiddleClose', 'GTI_Right']

cars_GTI_trains = []
cars_GTI_tests = []
for cars_GTI_folder in cars_GTI_folders:
    cars_GTI_paths = glob.glob(cfg.join_path(vehicle_path, cars_GTI_folder + '/*.png'))
    split = int(len(cars_GTI_paths) * test_size)
    cars_GTI_tests.extend(cars_GTI_paths[0:split])
    cars_GTI_trains.extend(cars_GTI_paths[split:])

cars_KITTI = glob.glob(cfg.join_path(vehicle_path, 'KITTI_extracted/*.png'))
non_vehicle_path = cfg.vehicle_detection['non-vehicles']
notcars = glob.glob(cfg.join_path(non_vehicle_path, '**/*.png'))

t = time.time()
cars_GTI_train_features = extract_features(cars_GTI_trains)
cars_GTI_test_features = extract_features(cars_GTI_tests)

car_KITTI_features = extract_features(cars_KITTI)
notcar_features = extract_features(notcars)

t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
X = np.vstack((cars_GTI_train_features, cars_GTI_test_features, car_KITTI_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(cars_GTI_train_features) + len(cars_GTI_test_features) + len(car_KITTI_features)),
               np.zeros(len(notcar_features))))

X_train_GTI = scaled_X[0: len(cars_GTI_train_features)]
y_train_GTI = y[0: len(cars_GTI_train_features)]
X_test_GTI = scaled_X[len(cars_GTI_train_features): len(cars_GTI_train_features) + len(cars_GTI_test_features)]
y_test_GTI = y[len(cars_GTI_train_features): len(cars_GTI_train_features) + len(cars_GTI_test_features)]

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X[len(cars_GTI_train_features) + len(cars_GTI_test_features):],
    y[len(cars_GTI_train_features) + len(cars_GTI_test_features):], test_size=0.2, random_state=rand_state)

X_train = np.vstack((X_train_GTI, X_train))
y_train = np.hstack((y_train_GTI, y_train))
X_test = np.vstack((X_test_GTI, X_test))
y_test = np.hstack((y_test_GTI, y_test))

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
