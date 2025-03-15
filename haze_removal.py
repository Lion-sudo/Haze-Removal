import os
os.environ["OPENCV_LOG_LEVEL"]="SILENT"
import cv2
import removers_factory


# Constants
# String messages
GREET_USER = "Hello! Welcome to the Haze Removal Program."
WRONG_INPUT_PATH_EXCEPTION = "Error: Couldn't read the image. Please check the path and try again."
EXIT_MSG = "\nExiting the program... Goodbye!"
ASK_FOR_PATH = "Please provide the path to the image you'd like to use: \n"
SUCCESS_MSG = ("\n###################################\n"
               "Haze-Free image saved successfully!\n"
               "###################################\n")
DESIRE_TO_CONTINUE = "Would you like to continue or exit?"
CONTINUE_OPTIONS = ("1: Continue using the current image \n"
                    "2: Load a new image \n"
                    "3: Exit the program ")
ASK_CONTINUE_INPUT = "Enter your choice: "
INVALID_CONTINUE_INPUT = ("\nInvalid choice! \nPlease select 1 to continue using the current image, "
                          "2 to load a new image, 3 to exit the program.")

# Variables
CONTINUE_CHOICE = "1"
LOAD_NEW_IMAGE_CHOICE = "2"
EXIT_CHOICE = "3"
CONTINUE_WORK = False
FINISHED_WORK = True
SHOULD_LOAD_IMAGE = True
DONT_LOAD_IMAGE = False


# Helper functions
def get_output_name(input_id, method_name):
    return f"image_{input_id}_haze_free_{method_name}.jpg"


def ask_if_should_continue():
    print(DESIRE_TO_CONTINUE)
    print(CONTINUE_OPTIONS)
    while True:
        choice = input(ASK_CONTINUE_INPUT)
        match choice:
            case _ if choice == CONTINUE_CHOICE:
                return DONT_LOAD_IMAGE, CONTINUE_WORK

            case _ if choice == LOAD_NEW_IMAGE_CHOICE:
                return SHOULD_LOAD_IMAGE, CONTINUE_WORK

            case _ if choice == EXIT_CHOICE:
                print(EXIT_MSG)
                return DONT_LOAD_IMAGE, FINISHED_WORK

            case _:
                print(INVALID_CONTINUE_INPUT)


def load_image():
    path = input(ASK_FOR_PATH)
    image = cv2.imread(path)
    if image is None:
        raise Exception(WRONG_INPUT_PATH_EXCEPTION)
    return image


# Main function
def main(image, input_id):
        remover = removers_factory.create_haze_remover()
        if not remover:
            return DONT_LOAD_IMAGE, FINISHED_WORK
        cv2.imwrite(get_output_name(input_id, remover.get_method_name()), remover.remove_haze(image))
        print(SUCCESS_MSG)
        return ask_if_should_continue()


if __name__ == "__main__":
    try:
        print(GREET_USER)
        should_load_image, is_finished = SHOULD_LOAD_IMAGE, CONTINUE_WORK
        image, input_id = None, 0
        while not is_finished:
            if should_load_image:
                image = load_image()
                input_id += 1
            should_load_image, is_finished = main(image, input_id)
    except Exception as exception:
        print(exception)
