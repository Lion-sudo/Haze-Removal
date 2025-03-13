import cv2
import numpy as np
from abstract_haze_remover import AbstractHazeRemover


# Constants
DEBUG_MODE = True  # Todo: Notice
DARK_CHANNEL_CALCULATION_WINDOW_SIZE = 15
TOP_PERCENT_FOR_ESTIMATION = 0.001
ESTIMATING_TRANSMISSION_MAP = True
OMEGA = 0.95  # Notice: In the Weighted Guided Filter they used 31/32 instead
GUIDED_FILTERING_METHOD = 0
WEIGHTED_GUIDED_FILTERING_METHOD = 1
GUIDED_WINDOW_RADIUS = 20
GUIDED_FILTER_EPSILON = 10**(-3)
RECOVERY_EPSILON = 0.1


class DCPRemover(AbstractHazeRemover):
    def __init__(self, image, method):
        self._image = image
        self._dark_channel = None
        self._atmospheric_light = None
        self._transmission_map = None
        self._refinement_method = method

    def remove_haze(self):
        self._calculate_dark_channel()
        self._estimate_atmospheric_light()
        self._estimate_transmission_map()
        self._refine_transmission_map()
        return self._recover_haze_free_image()


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


    def _guided_filtering(self):
        # use the grayscale input image as the guidance image
        I = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        I = I.astype(np.float64)
        I = I / 255.0

        # use the original transmission map to be the filter input
        p = self._transmission_map

        # step 1 of the algorithm
        mean_I = cv2.boxFilter(I, ddepth=-1, ksize=(GUIDED_WINDOW_RADIUS, GUIDED_WINDOW_RADIUS))
        mean_p = cv2.boxFilter(p, ddepth=-1, ksize=(GUIDED_WINDOW_RADIUS, GUIDED_WINDOW_RADIUS))
        mean_Ip = cv2.boxFilter(I * p, ddepth=-1, ksize=(GUIDED_WINDOW_RADIUS, GUIDED_WINDOW_RADIUS))

        # step 2
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = cv2.boxFilter(I * I, ddepth=-1, ksize=(GUIDED_WINDOW_RADIUS, GUIDED_WINDOW_RADIUS))
        var_I = mean_II - mean_I * mean_I

        # step 3
        a = cov_Ip / (var_I + GUIDED_FILTER_EPSILON)
        b = mean_p - a * mean_I

        # step 4
        mean_a = cv2.boxFilter(a, ddepth=-1, ksize=(GUIDED_WINDOW_RADIUS, GUIDED_WINDOW_RADIUS))
        mean_b = cv2.boxFilter(b, ddepth=-1, ksize=(GUIDED_WINDOW_RADIUS, GUIDED_WINDOW_RADIUS))

        # step 5, save the refined transmission map
        self._transmission_map = mean_a * I + mean_b


    def _weighted_guided_filtering(self):
        pass


    def _refine_transmission_map(self):
        if self._refinement_method == GUIDED_FILTERING_METHOD:
            self._guided_filtering()
        else:
            self._weighted_guided_filtering()

        if DEBUG_MODE:
            cv2.imshow("Refined Transmission Map", self._transmission_map)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def _recover_haze_free_image(self):
        # convert the input image and the atmospheric light to float64 type
        self._image = self._image.astype(np.float64)
        self._atmospheric_light = self._atmospheric_light.astype(np.float64)

        # make sure we keep a some of the haze in very dense haze areas
        self._transmission_map = np.clip(self._transmission_map, RECOVERY_EPSILON, 1.0)
        self._transmission_map = np.expand_dims(self._transmission_map, axis=2)  # allow RGB channels multiplication

        # recover the haze using equation 22 from the paper
        haze_free_img = ((self._image - self._atmospheric_light) / self._transmission_map) + self._atmospheric_light

        # move back to 0-255 values and to uint8 type
        haze_free_img = np.clip(haze_free_img, 0, 255).astype(np.uint8)

        if DEBUG_MODE:
            cv2.imshow("Haze-Free Image", haze_free_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return haze_free_img



# Testing
image = cv2.imread("testing.jpg")
testing = DCPRemover(image, GUIDED_FILTERING_METHOD)
haze_free = testing.remove_haze()
cv2.imwrite(f"using {GUIDED_WINDOW_RADIUS} radius, {GUIDED_FILTER_EPSILON} epsilon.jpg", haze_free)


