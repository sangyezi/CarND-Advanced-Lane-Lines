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

vehicle_detection = {
    'vehicles': return_if_exist(join_path(base_dir, 'resources', 'images', 'vehicles')),
    'non-vehicles': return_if_exist(join_path(base_dir, 'resources', 'images', 'non-vehicles')),
    'color-space': 'YUV',  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    'orient': 12,
    'pix_per_cell': 8,
    'cell_per_block': 2,
    'hog_channel': 'ALL',  # Can be 0, 1, 2, or "ALL",
    'spatial_size': None,
    'hist_bins': None,
    'svc_file': "svc_pickle.p"
}

video_path = {
    'videos': return_if_exist(join_path(base_dir, 'resources', 'videos', 'input_videos')),
    'frames': create_if_not_exist(join_path(base_dir, 'resources', 'images', 'video_frames')),
    'output_videos': create_if_not_exist(join_path(base_dir, 'resources', 'videos', 'output_videos')),

}

