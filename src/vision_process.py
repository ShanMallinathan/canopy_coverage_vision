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
        self.marker_centers=[]

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
        self.corners=np.squeeze(self.corners)
        
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
        for i in range(np.shape(self.corners)[0]): #loop for each marker (usually 4)
            self.marker_centers.append(self.corners[i].mean(axis=0))
        self.marker_centers_numpy=np.array(self.marker_centers, np.float32)

    def warp_image(self):
        """
        Use the corners from the compute markers centers and use perspective transforms and warp the image
        ---
        Computes
        A numpy array representing the warped image
        """
        self.compute_marker_center()
        warped_points=np.array([[827,1169], [0, 1169], [827, 0], [0,0]], np.float32)
        transform=cv2.getPerspectiveTransform(self.marker_centers_numpy, warped_points)
        self.warped_img=cv2.warpPerspective(self.img, transform, (827, 1169))

vis=VisionProcess('media/new_aruco.png')
vis.warp_image()
cv2.imshow("warped", vis.warped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()