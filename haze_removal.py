import os
os.environ["OPENCV_LOG_LEVEL"]="SILENT"
import cv2
import removers_factory

# Constants
GREET_USER = "Hello! Welcome to the Haze Removal Program."
IMAGE_PATH = "city_haze.jpg"
WRONG_INPUT_PATH_EXCEPTION = "Error: Couldn't read the image. Please check the path and try again."
EXIT_MSG = "Exiting the program... Goodbye!"

def main(image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise Exception(WRONG_INPUT_PATH_EXCEPTION)
        remover = removers_factory.create_haze_remover()
        if not remover:
            return None
        return remover.remove_haze(image)

if __name__ == "__main__":
    print(GREET_USER)
    haze_free = main(IMAGE_PATH)
    if haze_free is not None:
        cv2.imwrite("haze_free.jpg", haze_free)
        print("Haze-free image saved successfully!")

