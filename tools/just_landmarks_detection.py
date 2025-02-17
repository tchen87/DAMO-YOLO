import sys
import os
from loguru import logger
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/model_zoo'))

from openvino import Core, get_version
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
import cv2


def LandmarkDetection(imgpath, bboxes):
    core = Core()
    landmarks_detector = LandmarksDetector(core, 'landmarks-regression-retail-0009.xml')
    landmarks_detector.deploy('CPU', 16)
    logger.debug('reading from {}' , imgpath)

    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    rois = []
    for bbox in bboxes :
        x0 = bbox[0]
        y0 = bbox[1]
        x1 = bbox[2]
        y1 = bbox[3]
        roiInput = [0, 1, 1, x0, y0, x1 - x0, y1 - y0]
        roi = FaceDetector.Result(roiInput)
        rois.append(roi)
        
    landmarks = landmarks_detector.infer((img, rois))
    size = img.shape[:2]
    logger.debug(rois.count)
    
    for i in range(len(rois)) :
        landmark = landmarks[i]
        roi = rois[i]
        
        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), size[1])
        ymax = min(int(roi.position[1] + roi.size[1]), size[0])

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)
        for point in landmark:
            x = xmin + (roi.size[0] * point[0])
            y = ymin + (roi.size[1] * point[1])

            cv2.circle(img, (int(x), int(y)), 1, (0, 255, 255), 2)
    
    justfilename = os.path.basename(imgpath)

    split = os.path.splitext(justfilename)
    outputpath = split[0] + 'Landmarks.png'
    outputpath = os.path.join('./outputs', outputpath)
    logger.debug('writing to {}' , outputpath)
    cv2.imwrite(outputpath, img[:, :, ::-1])      
    


def main():
    LandmarkDetection('test.png', 310, 228, 352, 291)  


if __name__ == '__main__':
    sys.exit(main() or 0)
