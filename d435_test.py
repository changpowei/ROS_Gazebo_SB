import rospy
from sensor_msgs.msg import LaserScan, Image
import numpy as np
import sys
import cv2
from cv_bridge import CvBridge, CvBridgeError

class depth_img():

    def __init__(self):
        self.viz_image_cv2 = True
        print("init")

    def getDepthImage(self):
        '''
        ROS callback function

        return depth image from the front camera
        '''
        while True:
            try:
                DepthImage = rospy.wait_for_message('/camera/depth/image_rect_raw', Image, timeout=5)
                Depth_img = self.Depthcallback(DepthImage)

                # print("Image Heigh:%d, Width:%d"%(img_h, img_w))
                return Depth_img
            except Exception as e:
                rospy.logfatal("Error to get depth image " + str(e))

    def imgmsg_to_cv2(self, img_msg):
        if img_msg.encoding != "16UC1":
            rospy.logerr(
                "This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
        dtype = np.dtype("uint16")  # Hardcode to 8 bits...
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 1),
                                  # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                                  dtype=dtype, buffer=img_msg.data)
        # If the byt order is different between the message and the system.
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            image_opencv = image_opencv.byteswap().newbyteorder()
        return image_opencv

    def normalize(self, image, max, min):

        image_normalize = (image - min) / (max - min)

        return image_normalize

    def Depthcallback(self, msg_depth):
        # cv_bridge = CvBridge()
        try:
            """
            # The depth image is a single-channel float32 image
            # the values is the distance in mm in z axis
            """
            image = np.frombuffer(msg_depth.data, dtype=np.uint16).reshape(msg_depth.height, msg_depth.width, -1)
            # image = cv_bridge.imgmsg_to_cv2(msg_depth, "32FC1")
            # image = self.imgmsg_to_cv2(msg_depth)

            # Resize to the desired size
            image_resized = cv2.resize(image, (96, 54), interpolation=cv2.INTER_LINEAR)[:48, :].astype(np.float32)

            # nan_idx = np.argwhere(np.isnan(image_resized))
            # max_val = np.nanmax(image_resized)
            # for i in range(len(nan_idx)):
            #     image_resized[nan_idx[i][0], nan_idx[i][1]] = max_val

            """
            # Convert the depth image to a Numpy array since most cv2 functions
            # require Numpy arrays.
            """
            # cv_image_array = np.array(image, dtype=np.dtype('f8'))

            """
            # Normalize the depth image to fall between 0 (black) and 1 (white)
            # http://docs.ros.org/electric/api/rosbag_video/html/bag__to__video_8cpp_source.html lines 95-125
            """
            # image_norm = cv2.normalize(image_resized, image_resized, 0, 1, cv2.NORM_MINMAX)
            image_norm = self.normalize(image_resized, max=10000, min=200)

            # self.depthimg = cv_image_resized
            if self.viz_image_cv2:
                cv2.imshow("Depth Image", image_norm)
                cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)
        return image_norm


if __name__ == '__main__':
    rospy.init_node("depth_image_processor")
    depth_img = depth_img()

    while True:
        depthImage = depth_img.getDepthImage()

        c = cv2.waitKey(7)
        if c == 27 or c == 13:
            break
