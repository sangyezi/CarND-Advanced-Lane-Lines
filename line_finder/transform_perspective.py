import pickle
import config as cfg
import cv2


class TransformPerspective:
    """
    Provide functinality to transform perspective images taken by car camera
    """
    def __init__(self):
        perspective_matrix_path = cfg.line_finder['perspective_matrix_file']

        try:
            dict_pickle = pickle.load(open(perspective_matrix_path, "rb"))
            self.perspective_matrix = dict_pickle['M']
            self.perspective_matrix_inv = dict_pickle['Minv']
        except Exception:
            self.perspective_matrix = None

    def transform(self, img):
        """
        transform image
        :param img: input image
        :return: transformed image
        """
        img_size = (img.shape[1], img.shape[0])
        transformed = cv2.warpPerspective(img, self.perspective_matrix, img_size, flags=cv2.INTER_LINEAR)
        return transformed

    def inverse_transform(self, img):
        """
        transform image back
        :param img: input - transformed image
        :return: inverse transformed image
        """
        img_size = (img.shape[1], img.shape[0])
        untransformed = cv2.warpPerspective(img, self.perspective_matrix_inv, img_size, flags=cv2.INTER_LINEAR)
        return untransformed