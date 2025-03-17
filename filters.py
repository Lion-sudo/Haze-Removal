import cv2
import numpy as np


# Constants
DEBUG_MODE = False
GUIDED_WINDOW_RADIUS = 40
GUIDED_FILTER_EPSILON = 0.001
WEIGHTED_EPSILON = 0.001
BFILTER_WIN_SIZE = 9
BFILTER_SIGMA_COLOR = 75
BFILTER_SIGMA_SPACE = 75
SOBEL_WIN_SIZE = 3
SIGMA_FOR_WEIGHTS = 0.1
WEIGHTED_R = 30


def guided_filtering(guidance_image, filter_input):
    # use the guidance image in grayscale
    I = cv2.cvtColor((guidance_image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    I = I.astype(np.float32) / 255.0

    # use the original transmission / depth map as the filter input
    p = filter_input

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

    # step 5, return the filtered output
    filtered_output = mean_a * I + mean_b
    return filtered_output


def weighted_guided_filtering(guidance_image, filter_input):
    # use the guidance image in grayscale
    I = cv2.cvtColor((guidance_image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # use the original transmission / depth map as the filter input
    p = filter_input

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

    # step 5, return the filtered output
    filtered_output = mean_a * I + mean_b
    return filtered_output
