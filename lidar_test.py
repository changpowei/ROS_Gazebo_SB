import rospy
from sensor_msgs.msg import LaserScan, Image
import numpy as np
import math
from geometry_msgs.msg import Pose2D, PoseStamped
import time
import threading
import tf

class odom():
    def __init__(self):
        self.old_robotX = None
        self.old_robotY = None
        self.old_yaw = None
        self.old_time = None
        self.old_targetX = None
        self.old_targetY = None
        print("Init")

    def getLaserData(self):
        '''
        ROS callback function

        return laser scan in 2D list
        '''
        while True:
            try:
                laserData = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                return laserData
            except Exception as e:
                rospy.logfatal("Error to get laser data " + str(e))

    def getOdomData(self):
        '''
        ROS callback function
        Modify odom data quaternion to euler

        return yaw, posX, posY of robot known as Pos2D
        '''
        listener = tf.TransformListener()
        while True:
            try:

                (trans, rot) = listener.lookupTransform('/map', '/laser', time=rospy.Duration(0.0))
                robotX = trans[0]
                robotY = trans[1]
                quatTuple = (
                    rot[0],
                    rot[1],
                    rot[2],
                    rot[3],
                )
                roll, pitch, yaw = tf.transformations.euler_from_quaternion(
                    quatTuple)

                now_time = time.time()
                run_time = now_time - start_time

                if self.old_time == None:
                    robot_linearVel = 0
                    robot_angularVel = 0

                else:
                    time_diff = run_time - self.old_time
                    robot_linearVel = math.sqrt((robotX - self.old_robotX)**2 + (robotY - self.old_robotY)**2) / time_diff
                    robot_angularVel = (yaw - self.old_yaw) / time_diff

                self.old_robotX = robotX
                self.old_robotY = robotY
                self.old_yaw = yaw
                self.old_time = run_time

                return yaw, robotX, robotY, robot_linearVel, robot_angularVel, run_time

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

    def calcHeadingAngle(self, targetPointX, targetPointY, yaw, robotX, robotY):
        '''
        Calculate heading angle from robot to target

        return angle in float
        '''
        # targetPointX = self.old_targetX
        # targetPointY = self.old_targetY

        targetAngle = math.atan2(targetPointY - robotY, targetPointX - robotX)

        heading = targetAngle - yaw
        if heading > math.pi:
            heading -= 2 * math.pi

        elif heading < -math.pi:
            heading += 2 * math.pi

        return round(heading, 2)

    def get_target(self):
        global glob_old_targetX
        global glob_old_targetY
        t = threading.currentThread()
        while getattr(t, "get_target", True):
            try:
                target_pose = rospy.wait_for_message(topic="/move_base_simple/goal", topic_type=PoseStamped, timeout=5)
                targetPointX = target_pose.pose.position.x
                targetPointY = target_pose.pose.position.y
                print("Reset the target to ({}, {})".format(targetPointX, targetPointY))
                glob_old_targetX = targetPointX
                glob_old_targetY = targetPointY
                # return targetPointX, targetPointY
            except Exception as e:
                if glob_old_targetX == None and glob_old_targetY == None:
                    # rospy.logfatal("Didn't set target yet! " + str(e))
                    print("Didn't set target yet! Set target to (0, 0)")
                    glob_old_targetX = 0
                    glob_old_targetY = 0

                # return self.old_targetX, self.old_targetY
        print("Stopping get_target thread.")



if __name__ == '__main__':
    rospy.init_node("odom_processor")

    glob_old_targetX = 0
    glob_old_targetY = 0

    odom = odom()
    start_time = time.time()

    t_get_target = threading.Thread(target=odom.get_target)
    # t_getOdomData = threading.Thread(target=odom.getOdomData)

    t_get_target.start()
    # t_getOdomData.start()

    while True:

        # targetPointX, targetPointY = odom.get_target()
        yaw, robotX, robotY, robot_linearVel, robot_angularVel, run_time = odom.getOdomData()

        target_heading_ang = odom.calcHeadingAngle(targetPointX = glob_old_targetX, targetPointY = glob_old_targetY,yaw = yaw, robotX = robotX, robotY = robotY)

        print("X:{}".format(robotX))
        print("Y:{}".format(robotY))
        print("Theta:{}".format(yaw))
        print("lidar_linearVel:{}".format(robot_linearVel))
        print("lidat_angularVel:{}".format(robot_angularVel))
        print("time:{}".format(run_time))
        print("heading_angle:{}\n".format(target_heading_ang))

        if run_time >= 180:
            print("Fuck")
            t_get_target.get_target = False
            break