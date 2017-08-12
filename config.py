import os


def join_path(*args):
    return os.path.abspath(os.path.join(*args))


def return_if_exist(input_path):
    if os.path.exists(input_path):
        return input_path


def create_if_not_exist(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path


base_dir = os.path.dirname(__file__)

camera_calibration = {
    'input': return_if_exist(join_path(base_dir, 'resources', 'images', 'camera_cal')),
    'output':  create_if_not_exist(join_path(base_dir, 'resources', 'images', 'camera_cal_corners')),
    'grid_rows': 6,
    'grid_columns': 9,
    'pickle_filename': return_if_exist(join_path(base_dir, 'camera_calibration', 'wide_dist_pickle.p'))
}

line_finder = {
    'input': return_if_exist(join_path(base_dir, 'resources', 'images', 'test_images')),
    'output': create_if_not_exist(join_path(base_dir, 'resources', 'images', 'output_images'))
}

video_path = {
    'videos': return_if_exist(join_path(base_dir, 'resources', 'videos')),
    'frames': create_if_not_exist(join_path(base_dir, 'resources', 'images', 'video_frames'))
}

