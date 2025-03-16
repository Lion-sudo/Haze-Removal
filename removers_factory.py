from hazeRemovers.DCP_remover import DCPRemover, GUIDED_FILTERING_METHOD, WEIGHTED_GUIDED_FILTERING_METHOD
from hazeRemovers.CAP_remover import CAPRemover


# Constants
INSTRUCTION = "\nChoose the haze removal method:"
OPTIONS = ("1: Dark Channel Prior using Guided Filter "
           "\n2: Dark Channel Prior using Weighted Guided Filter "
           "\n3: Color Attenuation Prior "
           "\n4: Exit the program")
ASK_INPUT = "Enter the number corresponding to your choice: "
INVALID_CHOICE = "\nInvalid choice!"
EXIT_MSG = "\nExiting the program... Goodbye!"


# Codes
DCP_GUIDED = "1"
DPC_WEIGHTED_GUIDED = "2"
CAP = "3"
EXIT_CHOICE = "4"


def print_options():
    print(INSTRUCTION)
    print(OPTIONS)


def create_haze_remover():
    while True:
        print_options()
        choice = input(ASK_INPUT)
        match choice:
            case _ if choice == DCP_GUIDED:
                return DCPRemover(GUIDED_FILTERING_METHOD)

            case _ if choice == DPC_WEIGHTED_GUIDED:
                return DCPRemover(WEIGHTED_GUIDED_FILTERING_METHOD)

            case _ if choice == CAP:
                return CAPRemover()

            case _ if choice == EXIT_CHOICE:
                print(EXIT_MSG)
                return None

            case _:
                print(INVALID_CHOICE)
