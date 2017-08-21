import cv2
import config as cfg
from camera_calibration.undistort_image import Undistort


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


def video_extract(video_path, output_path, video_name, transform):
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
            cv2.imwrite("%s/%s_frame%d.jpg" % (output_path, video_name, count), image)  # save frame as JPEG file


def main():
    video_name = 'project_video'

    video_path = cfg.join_path(cfg.video_path['videos'], video_name + '.mp4')
    output_path = cfg.create_if_not_exist(cfg.join_path(cfg.video_path['frames'], video_name))

    undistort = Undistort().undistort_image

    video_extract(video_path, output_path, video_name, undistort)

if __name__ == '__main__':
    main()
