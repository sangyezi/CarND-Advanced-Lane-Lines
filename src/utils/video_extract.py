import cv2
import os
import pickle


def video_frame_generator(video_path):
    """
    read video from video path, and yield its frames once at a time
    :param video_path: video path
    :return: frame count and frame image
    """
    vidcap = cv2.VideoCapture(video_path)
    count = 1

    success = True
    while success:
        success, image = vidcap.read()
        yield count, image
        count += 1


def video_extract(video_path, output_path, transform):
    """
    extract frames from video, transform them, and write them as image files to output path
    :param video_path: video path
    :param output_path: output path to write images
    :param transform: functor to transform frame image
    :return:
    reference: https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
    """
    for count, image in video_frame_generator(video_path):
        if transform is not None and image is not None:
            image = transform(image)
        if image is not None:
            cv2.imwrite("%s/frame%d.jpg" % (output_path, count), image)  # save frame as JPEG file


def main():
    base_dir = os.path.dirname(__file__)
    video_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'project_video.mp4'))

    output_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'project_video_frames_undistorted'))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    camera_pickle_path = os.path.abspath(os.path.join(base_dir, '..', 'camera_calibration/wide_dist_pickle.p'))

    dist_pickle = pickle.load(open(camera_pickle_path, "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    undistort = lambda img: cv2.undistort(img, mtx, dist, None, mtx)

    video_extract(video_path, output_path, undistort)

if __name__ == '__main__':
    main()
