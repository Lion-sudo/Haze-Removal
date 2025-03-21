import cv2
import numpy as np
from hazeRemovers.abstract_haze_remover import AbstractHazeRemover
from filters import guided_filtering, weighted_guided_filtering


# Constants
DARK_CHANNEL_CALCULATION_WINDOW_SIZE = 15
TOP_PERCENT_FOR_ESTIMATION = 0.001
OMEGA = 0.85
GUIDED_FILTERING_METHOD = 0
WEIGHTED_GUIDED_FILTERING_METHOD = 1
RECOVERY_EPSILON = 0.15
GUIDED_METHOD_STR = "DCP_Guided_Method"
WEIGHTED_METHOD_STR = "DCP_Weighted_Method"


class DCPRemover(AbstractHazeRemover):
    def __init__(self, method):
        self.__image = None
        self.__dark_channel = None
        self.__atmospheric_light = None
        self.__transmission_map = None
        self.__refinement_method = method


    def get_method_name(self):
        return WEIGHTED_METHOD_STR if self.__refinement_method else GUIDED_METHOD_STR


    def remove_haze(self, image):
        self.__set_image(image)
        self.__calculate_dark_channel()
        self.__estimate_atmospheric_light()
        self.__estimate_transmission_map()
        self.__refine_transmission_map()
        return self.__recover_haze_free_image()


    def __set_image(self, image):
        self.__image = image.astype(np.float32) / 255.0 # normalize to [0, 1]


    def __calculate_dark_channel(self, estimating_transmission_map=False):
        if not estimating_transmission_map:
            # get the minimal intensity from the RGB channels to compute the dark channel
            min_channel = np.min(self.__image, axis=2)
        else:
            # get the minimal intensity from the RGB channels of the normalized image
            # in order to compute the transmission map
            normalized_image = self.__image / self.__atmospheric_light
            min_channel = np.min(normalized_image, axis=2)

        # create a min kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (DARK_CHANNEL_CALCULATION_WINDOW_SIZE,
                                                            DARK_CHANNEL_CALCULATION_WINDOW_SIZE))

        if not estimating_transmission_map:
            # apply the kernel and save the dark channel
            self.__dark_channel = cv2.erode(min_channel, kernel)
        else:
            # apply the kernel and save the transmission map
            self.__transmission_map = 1 - OMEGA * (cv2.erode(min_channel, kernel))


    def __estimate_atmospheric_light(self):
        # flat the dark channel
        flat_dark = self.__dark_channel.flatten()

        # calculate the amount of pixels we need to look at (minimum 1 pixel)
        num_top_pixels = max(int(len(flat_dark) * TOP_PERCENT_FOR_ESTIMATION), 1)

        # get indices of the brightest pixels in the dark channel (symbolizing haze)
        top_indices = np.argpartition(flat_dark, -num_top_pixels)[-num_top_pixels:]

        # convert indices to 2D coordinates
        rows, cols = self.__dark_channel.shape
        top_coords = np.unravel_index(top_indices, (rows, cols))

        # get the RGB channels from the input image
        top_pixels = self.__image[top_coords]

        # use the average of top brightest pixels as the atmospheric light
        self.__atmospheric_light = np.mean(top_pixels, axis=0)


    def __estimate_transmission_map(self):
        self.__calculate_dark_channel(estimating_transmission_map=True)


    def __refine_transmission_map(self):
        if self.__refinement_method == GUIDED_FILTERING_METHOD:
            self.__transmission_map = guided_filtering(self.__image, self.__transmission_map)
        else:
            self.__transmission_map = weighted_guided_filtering(self.__image, self.__transmission_map)


    def __recover_haze_free_image(self):
        # make sure we keep a some of the haze in very dense haze areas
        self.__transmission_map = np.clip(self.__transmission_map, RECOVERY_EPSILON, 1.0)
        self.__transmission_map = np.expand_dims(self.__transmission_map, axis=2)  # allow RGB calculations

        # recover the haze using equation 22 from the paper
        haze_free_img = ((self.__image - self.__atmospheric_light) / self.__transmission_map) + self.__atmospheric_light

        # convert back to 0-255 and to uint8
        haze_free_img = np.clip(haze_free_img * 255, 0, 255).astype(np.uint8)
        return haze_free_img
