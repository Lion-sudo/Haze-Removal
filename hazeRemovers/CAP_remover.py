import cv2
import numpy as np
from hazeRemovers.abstract_haze_remover import AbstractHazeRemover
from filters import guided_filtering, weighted_guided_filtering


# Constants
DEBUG_MODE = False
MIN_TRANSMISSION = 0.1
MAX_TRANSMISSION = 0.9
GUIDED_METHOD_NAME = "CAP_Guided_Method"
WEIGHTED_METHOD_NAME = "CAP_Weighted_Guided_Method"
DEPTH_THRESHOLD = 99.9
PAPER_THETA0 = 0.121779
PAPER_THETA1 = 0.959710
PAPER_THETA2 = -0.780245
PAPER_BETA = 1.0
MIN_WINDOW_SIZE = 15
GUIDED_FILTERING_METHOD = 0
WEIGHTED_GUIDED_FILTERING_METHOD = 1


class CAPRemover(AbstractHazeRemover):
    def __init__(self):
        self.__image = None
        self.__depth_map = None
        self.__atmospheric_light = None
        self.__transmission_map = None


    def get_method_name(self):
        return GUIDED_METHOD_NAME


    def remove_haze(self, image):
        self.__set_image(image)
        self.__calculate_depth_map()
        self.__estimate_atmospheric_light()
        self.__calculate_transmission_map()
        return self.__recover_haze_free_image()


    def __set_image(self, image):
        self.__image = image.astype(np.float32) / 255.0  # move to [0, 1]


    def __calculate_depth_map(self):
        # convert to HSV color space
        hsv = cv2.cvtColor((self.__image * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        # normalize s, v channels to [0,1]
        h, s, v = cv2.split(hsv)
        s = s / 255.0
        v = v / 255.0

        # calculate depth using the linear model from the paper (8)
        self.__depth_map = PAPER_THETA0 + PAPER_THETA1 * v - PAPER_THETA2 * s

        # use neighbor pixels to determine the depth map
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MIN_WINDOW_SIZE, MIN_WINDOW_SIZE))
        self.__depth_map = cv2.erode(self.__depth_map, kernel)

        # refine the depth map using the guided filter
        self.__depth_map = guided_filtering(self.__image, self.__depth_map)

        if DEBUG_MODE:
            # normalize map for better visualization
            depth_normalized = cv2.normalize(self.__depth_map, None, 0, 1, cv2.NORM_MINMAX)
            cv2.imshow("Depth Map", depth_normalized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def __estimate_atmospheric_light(self):
        # set a threshold for depth
        depth_threshold = np.percentile(self.__depth_map, DEPTH_THRESHOLD)

        # create mask of pixels where depth exceeds the threshold
        mask = self.__depth_map > depth_threshold

        # use the pixels where depth exceeds threshold
        self.__atmospheric_light = np.mean(self.__image[mask], axis=0)

        if DEBUG_MODE:
            print(self.__atmospheric_light)


    def __calculate_transmission_map(self):
        # calculate the transmission map using equation (2) in the paper
        self.__transmission_map = np.exp(-PAPER_BETA * self.__depth_map)

        if DEBUG_MODE:
            cv2.imshow("Transmission Map", self.__transmission_map)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def __recover_haze_free_image(self):
        # clip and expand the transmission map
        self.__transmission_map = np.clip(self.__transmission_map, MIN_TRANSMISSION, MAX_TRANSMISSION)
        self.__transmission_map = np.expand_dims(self.__transmission_map, axis=2)  # allow RGB calculations

        # use equation (23) to get the haze free image
        haze_free_img = ((self.__image - self.__atmospheric_light) / self.__transmission_map) + self.__atmospheric_light

        # convert to [0-255] and uint8
        haze_free_img = np.clip(haze_free_img * 255, 0, 255).astype(np.uint8)

        if DEBUG_MODE:
            cv2.imshow("Haze-Free Image", haze_free_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return haze_free_img
