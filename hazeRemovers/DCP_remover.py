import cv2
import numpy as np
from abstract_haze_remover import AbstractHazeRemover


# Constants
DEBUG_MODE = False
DARK_CHANNEL_CALCULATION_WINDOW_SIZE = 15
TOP_PERCENT_FOR_ESTIMATION = 0.001
OMEGA = 0.85  # got better results than 0.95
GUIDED_FILTERING_METHOD = 0
WEIGHTED_GUIDED_FILTERING_METHOD = 1
GUIDED_WINDOW_RADIUS = 60
GUIDED_FILTER_EPSILON = 0.001
WEIGHTED_R = 30  # R - radius
WEIGHTED_EPSILON = 0.001
RECOVERY_EPSILON = 0.15
SOBEL_WIN_SIZE = 3
SIGMA_FOR_WEIGHTS = 0.1
BFILTER_WIN_SIZE = 9
BFILTER_SIGMA_COLOR = 75
BFILTER_SIGMA_SPACE = 75


class DCPRemover(AbstractHazeRemover):
    def __init__(self, image, method):
        self._image = image.astype(np.float32) / 255.0  # normalize to [0, 1]
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

        # get indices of the brightest pixels in the dark channel (symbolizing haze)
        top_indices = np.argpartition(flat_dark, -num_top_pixels)[-num_top_pixels:]

        # convert indices to 2D coordinates
        rows, cols = self._dark_channel.shape
        top_coords = np.unravel_index(top_indices, (rows, cols))

        # get the RGB channels from the input image
        top_pixels = self._image[top_coords]

        # use the average of top brightest pixels as the atmospheric light
        self._atmospheric_light = np.mean(top_pixels, axis=0)

        if DEBUG_MODE:
            print("Atmospheric Light:", self._atmospheric_light)
            # Create a copy for visualization that's in 0-255 range
            image_with_box = (self._image * 255).astype(np.uint8)
            min_row, max_row = np.min(top_coords[0]), np.max(top_coords[0])
            min_col, max_col = np.min(top_coords[1]), np.max(top_coords[1])
            cv2.rectangle(image_with_box, (min_col, min_row), (max_col, max_row), (0, 255, 0), 2)
            cv2.imshow("Atmospheric Light Region", image_with_box)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def _estimate_transmission_map(self):
        self._calculate_dark_channel(estimating_transmission_map=True)


    def _guided_filtering(self):
        # use the grayscale input image as the guidance image
        I = cv2.cvtColor((self._image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        I = I.astype(np.float32) / 255.0

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
        # use the grayscale input image as the guidance image
        I = cv2.cvtColor((self._image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        p = self._transmission_map

        # for better edge detection apply bilateral filter (less noise)
        I_smooth = cv2.bilateralFilter(I, BFILTER_WIN_SIZE, BFILTER_SIGMA_COLOR, BFILTER_SIGMA_SPACE)

        # edge detection using gradient magnitude
        grad_x = cv2.Sobel(I_smooth, cv2.CV_32F, 1, 0, ksize=SOBEL_WIN_SIZE)
        grad_y = cv2.Sobel(I_smooth, cv2.CV_32F, 0, 1, ksize=SOBEL_WIN_SIZE)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # normalize gradient for better weight calculation
        gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude)

        # get weights based on gradient (lower weight near edges to preserve them)
        weights = np.exp(-(gradient_magnitude ** 2) / (2 * SIGMA_FOR_WEIGHTS ** 2))

        if DEBUG_MODE:
            cv2.imshow("Edge Weights", weights)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # weighted box filters
        N = cv2.boxFilter(np.ones_like(I), ddepth=-1, ksize=(WEIGHTED_R, WEIGHTED_R))
        sum_weights = cv2.boxFilter(weights, ddepth=-1, ksize=(WEIGHTED_R, WEIGHTED_R))

        # step 1
        mean_I = cv2.boxFilter(I * weights, ddepth=-1, ksize=(WEIGHTED_R, WEIGHTED_R)) / (sum_weights + 1e-6)
        mean_p = cv2.boxFilter(p * weights, ddepth=-1, ksize=(WEIGHTED_R, WEIGHTED_R)) / (sum_weights + 1e-6)
        mean_Ip = cv2.boxFilter(I * p * weights, ddepth=-1, ksize=(WEIGHTED_R, WEIGHTED_R)) / (sum_weights + 1e-6)
        mean_II = cv2.boxFilter(I * I * weights, ddepth=-1, ksize=(WEIGHTED_R, WEIGHTED_R)) / (sum_weights + 1e-6)

        # step 2
        var_I = mean_II - mean_I * mean_I
        cov_Ip = mean_Ip - mean_I * mean_p

        # step 3
        a = cov_Ip / (var_I + WEIGHTED_EPSILON)
        b = mean_p - a * mean_I

        # step 4
        mean_a = cv2.boxFilter(a, ddepth=-1, ksize=(WEIGHTED_R, WEIGHTED_R)) / N
        mean_b = cv2.boxFilter(b, ddepth=-1, ksize=(WEIGHTED_R, WEIGHTED_R)) / N

        # step 5, save the refined transmission map
        self._transmission_map = mean_a * I + mean_b


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
        # make sure we keep a some of the haze in very dense haze areas
        self._transmission_map = np.clip(self._transmission_map, RECOVERY_EPSILON, 1.0)
        self._transmission_map = np.expand_dims(self._transmission_map, axis=2)  # allow RGB channels multiplication

        # recover the haze using equation 22 from the paper
        haze_free_img = ((self._image - self._atmospheric_light) / self._transmission_map) + self._atmospheric_light

        # convert back to 0-255 and to uint8
        haze_free_img = np.clip(haze_free_img * 255, 0, 255).astype(np.uint8)

        if DEBUG_MODE:
            cv2.imshow("Haze-Free Image", haze_free_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return haze_free_img


# Testing
def test_methods(image_path, output_path):
    image = cv2.imread(image_path)

    for method in [GUIDED_FILTERING_METHOD, WEIGHTED_GUIDED_FILTERING_METHOD]:
        method_name = "guided" if method == GUIDED_FILTERING_METHOD else "weighted_guided"
        remover = DCPRemover(image, method)
        haze_free = remover.remove_haze()
        output_name = output_path.replace('.jpg', f'_{method_name}.jpg')
        cv2.imwrite(output_name, haze_free)


test_methods("city_haze.jpg", "dehazed_result.jpg")