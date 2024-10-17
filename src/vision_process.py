import cv2
import numpy as np

class VisionProcess():
    """
    class to detect aruco markers within the image, warp the image and return the coordinartes for SAM2 
    """
    def __init__(self, image_loc) -> None:
        self.img=cv2.imread(image_loc)
        self.gray=cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.aruco_dict_class=cv2.aruco.DICT_6X6_50

    def detect_aruco_markers(self):
        """
        Function to detect aruco markers from a grayscale input image 
        ---
        Returns 
        corners - 4 corners of a marker
        id - Aruco ID

        """
        aruco_dict=cv2.aruco.getPredefinedDictionary(self.aruco_dict_class)
        det_params=cv2.aruco.DetectorParameters()
        detector=cv2.aruco.ArucoDetector(aruco_dict, det_params)
        self.corners, self.id, rejected = detector.detectMarkers(self.gray)
        
    def compute_marker_center(self):
        """
        Function to compute the center of eacj markers
        ---
        Requires
        corners from detect_aruco_markers function

        ---
        Returns
        center of each marker
        """
        self.detect_aruco_markers()
        
