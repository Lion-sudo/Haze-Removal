import cv2
import numpy as np
from abstract_haze_remover import AbstractHazeRemover


# Constants
DEBUG_MODE = True  # Todo: Notice
DARK_CHANNEL_CALCULATION_WINDOW_SIZE = 15
TOP_PERCENT_FOR_ESTIMATION = 0.001
ESTIMATING_TRANSMISSION_MAP = True
OMEGA = 0.95


class DCPRemover(AbstractHazeRemover):
    def __init__(self, image):
        self._image = image
        self._dark_channel = None
        self._atmospheric_light = None
        self._transmission_map = None

    def remove_haze(self):
        self._calculate_dark_channel()
        self._estimate_atmospheric_light()
        self._estimate_transmission_map()
        self._soft_matt()

    def _calculate_dark_channel(self, estimating_transmission_map=False):
        if not estimating_transmission_map:
            # get the minimal intensity from the RGB channels to compute the dark channel
            min_channel = np.min(self._image, axis=2)
        else:
            # get the minimal intensity from the RGB channels of the normalized image
            # in order to compute the transmission map
            normalized_image = self._image / self._atmospheric_light
            min_channel = np.min(normalized_image, axis=2)

        # create a min kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (DARK_CHANNEL_CALCULATION_WINDOW_SIZE,
                                                            DARK_CHANNEL_CALCULATION_WINDOW_SIZE))

        if not estimating_transmission_map:
            # apply the kernel and save the dark channel
            self._dark_channel = cv2.erode(min_channel, kernel)
        else:
            # apply the kernel and save the transmission map
            self._transmission_map = 1 - OMEGA * (cv2.erode(min_channel, kernel))

        if DEBUG_MODE:
            if not estimating_transmission_map:
                cv2.imshow("Dark Channel", self._dark_channel)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                cv2.imshow("Transmission Map", self._transmission_map)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


    def _estimate_atmospheric_light(self):
        # flat the dark channel
        flat_dark = self._dark_channel.flatten()

        # calculate the amount of pixels we need to look at (minimum 1 pixel)
        num_top_pixels = max(int(len(flat_dark) * TOP_PERCENT_FOR_ESTIMATION), 1)

        # get indices of the 0.1% brightest pixels in the dark channel (symbolizing haze)
        top_indices = np.argpartition(flat_dark, -num_top_pixels)[-num_top_pixels:]

        # convert indices to 2D coordinates
        rows, cols = self._dark_channel.shape
        top_coords = np.unravel_index(top_indices, (rows, cols))

        # get the RGB channels from the input image
        top_pixels = self._image[top_coords]

        # get the pixel with the max intensity (sum of RGB channels)
        # TODO: Check if average will work better on results
        self._atmospheric_light = top_pixels[np.argmax(np.sum(top_pixels, axis=1))]

        if DEBUG_MODE:
            print(self._atmospheric_light)
            min_row, max_row = np.min(top_coords[0]), np.max(top_coords[0])
            min_col, max_col = np.min(top_coords[1]), np.max(top_coords[1])
            image_with_box = image.copy()
            cv2.rectangle(image_with_box, (min_col, min_row), (max_col, max_row), (0, 255, 0), 2)
            cv2.imshow("Check box", image_with_box)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def _estimate_transmission_map(self):
        self._calculate_dark_channel(ESTIMATING_TRANSMISSION_MAP)


    def _soft_matt(self, lambda_=1e-4):
        pass


# Testing
image = cv2.imread("testing.jpg")
testing = DCPRemover(image)
testing.remove_haze()


