from tools.demo import *
import os

from tools.just_landmarks_detection import LandmarkDetection

def list_files_in_directory(directory_path):
    files = []
    # List all files in the directory as strings
    for f in os.listdir(directory_path) :
        file = os.path.join(directory_path, f)
        if os.path.isfile(file) :
            files.append(file)

    return files


def main():
    #test()
    files = list_files_in_directory('datasets/training_data_03112025/')
    for file in files:
        logger.debug('file = {}', file)
        outputBBoxes = RunInferenceOnImage(file)
        logger.debug(outputBBoxes)
        logger.debug('file = {}', file)
        # break
        LandmarkDetection(file, outputBBoxes)



if __name__ == '__main__':
    main()
