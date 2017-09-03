import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from vehicle_detection.image_features import convert_color, get_hog_features, bin_spatial, color_hist, normalize_img
import config as cfg
from scipy.ndimage.measurements import label


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, cspace, hog_channel, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    rectangles = []
    windows = []

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

    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

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

            xbox_left = np.int(xleft * scale)
            ytop_draw = np.int(ytop * scale)
            win_draw = np.int(window * scale)

            windows.append(((xbox_left, ytop_draw + ystart), (xbox_left+win_draw, ytop_draw + win_draw+ystart)))
            if test_prediction == 1:
                rectangles.append(((xbox_left, ytop_draw + ystart), (xbox_left+win_draw, ytop_draw + win_draw+ystart)))

    return windows, rectangles


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 4)
    # Return the image
    return img


def car_multiple_detections(img, draw=False, save_path=None):
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

    yranges_scales = [[400, 500, 1.2], [400, 656, 1.5], [400, 720, 2],  [400, 720, 2.5]]

    all_rectangles = []

    img_rectangles = np.copy(img)
    img_windows = np.copy(img)

    for ystart, ystop, scale, in yranges_scales:
        windows, rectangles = find_cars(img, ystart, ystop, scale, svc, X_scaler, cspace, hog_channel, orient, pix_per_cell,
                           cell_per_block, spatial_size, hist_bins)
        all_rectangles.extend(rectangles)

        color = (np.random.randint(0, 8) * 32, np.random.randint(0, 8) * 32, np.random.randint(0, 8) * 32)
        for window in [windows[0], windows[1], windows[-2], windows[-1]]:
            cv2.rectangle(img_windows, window[0], window[1], color, 6)
        for rectangle in rectangles:
            cv2.rectangle(img_rectangles, rectangle[0], rectangle[1], color, 6)

    heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, all_rectangles)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    if draw:
        print(labels[1], 'cars found')

        fig = plt.figure(figsize=(15, 10))
        plt.subplot(321)
        plt.imshow(img)
        plt.title('Original image', fontsize=18)
        plt.subplot(322)
        plt.imshow(img_windows)
        plt.title('Windows (only first two and last two of a scale showed)', fontsize=18)
        plt.subplot(323)
        plt.imshow(img_rectangles)
        plt.title('Windows with car', fontsize=18)
        plt.subplot(324)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map', fontsize=18)
        plt.subplot(325)
        plt.imshow(labels[0], cmap='gray')
        plt.title('Labels', fontsize=18)
        plt.subplot(326)
        plt.imshow(draw_img)
        plt.title('Car Positions', fontsize=18)
        fig.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    return draw_img


def test():
    image_name = 'test5'
    image_path = cfg.join_path(cfg.line_finder['input'], image_name + '.jpg')
    img = mpimg.imread(image_path)

    # frame_name = 'project_video_frame500'
    # frame_path = cfg.join_path(cfg.video_path['frames'], 'project_video', frame_name + '.jpg')
    # img = mpimg.imread(frame_path)
    save_path = cfg.join_path(cfg.vehicle_detection['output'], image_name + '_car_finder.jpg')
    car_multiple_detections(img, True, save_path)


def project_video_process():
    frame = 0

    def process_image(image):
        """
        process image (identify the lane line), return the processed image
        :param image: input image
        :return: the processed image
        """
        nonlocal frame
        frame += 1
        text = 'frame: %d' % frame
        detected_img = car_multiple_detections(image, False)
        cv2.putText(detected_img, text, (10, detected_img.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return detected_img

    from moviepy.editor import VideoFileClip

    video_name = 'project_video.mp4'
    # video_name = 'challenge_video.mp4'
    # video_name = 'harder_challenge_video.mp4'

    name, ext = video_name.split('.')
    input_path = cfg.join_path(cfg.video_path['videos'], video_name)
    output_path = cfg.join_path(cfg.video_path['output_videos'], name + '_vehicle_detected.' + ext)

    # To speed up the testing process, only process a subclip of the first 5 seconds
    # clip1 = VideoFileClip(input_path).subclip(20, 22)
    clip1 = VideoFileClip(input_path)
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(output_path, audio=False)


if __name__ == '__main__':
    test()
    # project_video_process()
