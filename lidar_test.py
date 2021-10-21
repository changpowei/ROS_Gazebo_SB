import rospy
from sensor_msgs.msg import LaserScan, Image
import numpy as np
import math
from geometry_msgs.msg import PoseStamped
import time
import threading
import tf
import copy

class odom():
    def __init__(self):

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

        global robotX
        global robotY
        global roll
        global pitch
        global yaw
        listener = tf.TransformListener()
        t = threading.currentThread()
        rate = rospy.Rate(10.0)
        while getattr(t, "getOdomData", True):
            try:
                (trans, rot) = listener.lookupTransform('/map', '/laser', rospy.Time(0))
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

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

            rate.sleep()

    def getVelData(self, pre_X, pre_Y, pre_yaw, pre_time):

        now_X = copy.copy(robotX)
        now_Y = copy.copy(robotY)
        now_yaw = copy.copy(yaw)

        now_time = time.time()
        run_time = now_time - start_time

        if pre_time == None:
            robot_linearVel = 0
            robot_angularVel = 0

        else:
            time_diff = run_time - pre_time
            robot_linearVel = math.sqrt((now_X - pre_X) ** 2 + (now_Y - pre_Y) ** 2) / time_diff
            robot_angularVel = (now_yaw - pre_yaw) / time_diff
            # print("Linear = {}, Angular = {}".format(robot_linearVel, robot_angularVel))

        return now_X, now_Y, now_yaw, robot_linearVel, robot_angularVel, run_time

    def calcHeadingAngle(self, targetPointX, targetPointY, now_yaw, now_X, now_Y):
        '''
        Calculate heading angle from robot to target

        return angle in float
        '''
        # targetPointX = self.old_targetX
        # targetPointY = self.old_targetY

        targetAngle = math.atan2(targetPointY - now_Y, targetPointX - now_X)

        heading = targetAngle - now_yaw
        if heading > math.pi:
            heading -= 2 * math.pi

        elif heading < -math.pi:
            heading += 2 * math.pi

        return round(heading, 2)

    def get_target(self):
        global glob_old_targetX
        global glob_old_targetY
        glob_old_targetX = None
        glob_old_targetY = None
        t1 = threading.currentThread()
        while getattr(t1, "get_target", True):
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

    odom = odom()

    t_getOdomData = threading.Thread(target=odom.getOdomData)
    t_getOdomData.start()

    t_get_target = threading.Thread(target=odom.get_target)
    t_get_target.start()

    start_time = time.time()

    pre_X = None
    pre_Y = None
    pre_yaw = None
    pre_time = None

    print("Wait 10 secs for launching all threading.")
    time.sleep(10)

    while True:

        now_X, now_Y, now_yaw, robot_linearVel, robot_angularVel, run_time = odom.getVelData(pre_X = pre_X, pre_Y = pre_Y,
                                                                                           pre_yaw = pre_yaw,
                                                                                           pre_time = pre_time)

        target_heading_ang = odom.calcHeadingAngle(targetPointX = glob_old_targetX, targetPointY = glob_old_targetY,
                                                   now_yaw = now_yaw, now_X = now_X, now_Y = now_Y)

        print("(X, Y):({:.3f}, {:.3f}), yaw:{:.3f}, linearVel:{:.5f}, angularVel:{:.5f}, heading_angle:{:.3f}, time:{:.3f}".format(now_X,
                                                                                                       now_Y,
                                                                                                       now_yaw,
                                                                                                       robot_linearVel,
                                                                                                       robot_angularVel,
                                                                                                       target_heading_ang,
                                                                                                       run_time))

        pre_X = copy.copy(now_X)
        pre_Y = copy.copy(now_Y)
        pre_yaw = copy.copy(now_yaw)
        pre_time = run_time

        if run_time >= 90:
            print("Fuck")
            t_get_target.get_target = False
            t_getOdomData.getOdomData = False
            break

        time.sleep(0.3)