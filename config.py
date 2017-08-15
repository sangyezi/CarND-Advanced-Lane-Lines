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
    'camera_calibration_file': join_path(base_dir, 'camera_calibration', 'camera_data.p')
}

line_finder = {
    'input': return_if_exist(join_path(base_dir, 'resources', 'images', 'test_images')),
    'output': create_if_not_exist(join_path(base_dir, 'resources', 'images', 'output_images')),
    'perspective_matrix_file': join_path(base_dir, 'line_finder', 'warp_matrix.p')
}

video_path = {
    'videos': return_if_exist(join_path(base_dir, 'resources', 'videos', 'input_videos')),
    'frames': create_if_not_exist(join_path(base_dir, 'resources', 'images', 'video_frames')),
    'output_videos': create_if_not_exist(join_path(base_dir, 'resources', 'videos', 'output_videos')),

}

