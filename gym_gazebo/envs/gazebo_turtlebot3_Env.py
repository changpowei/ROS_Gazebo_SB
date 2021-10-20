import rospy
import roslaunch
import time
import numpy as np
import math
import sys
import random
import matplotlib.path as mpltPath
from shapely.geometry import Polygon
from shapely.geometry import Point as pot

from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState, GetModelState, GetModelStateRequest

from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point

from nav_msgs.msg import Odometry
import tf
from sensor_msgs.msg import LaserScan, Image, CompressedImage
from std_srvs.srv import Empty
from cv_bridge import CvBridge, CvBridgeError
import cv2
import gym
from gym import spaces


class gazebo_turtlebot3_Env(gym.Env):
    '''
    Main Gazebo environment class
    Contains reset and step function
    '''

    def __init__(self, config):
        # Initialize the node
        rospy.init_node('turtlebot3_gym_env', anonymous=True)

        # Connect to gazebo
        self.velPub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy(
            '/gazebo/reset_simulation', Empty)

        self.map_name = config.map_name
        self.viz_image_cv2 = config.viz_image_cv2
        self.random_target = config.random_target

        self.laserPointCount = 360  # 360 laser point in one time
        self.minCrashRange = 0.18  # Asume crash below this distance(front side lidar) 0.15
        self.front_side_lidar = list(range(24)) + list(
            range(360 - 24, 360, 1))  # front side lidar is using minCrashRange to consider crash [0~23, 336~359]
        self.minCrashRange_side = 0.18  # Asume crash below this distance(other sides lidar) 0.165
        self.laserMinRange = 0.12  # Modify laser data and fix min range to 0.12m
        self.laserMaxRange = 3.5  # Modify laser data and fix max range to 3.5m

        self.state_image = np.zeros((48, 96), dtype=np.float)
        self.state_sensor = np.zeros((self.laserPointCount + 4), dtype=np.float64)
        # self.stateSize = self.laserPointCount + 4  # Laser(arr), heading, distance, obstacleMinRange, obstacleAngle
        # self.observation_space = [self.state_image.shape, self.state_sensor.shape]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4972,), dtype=np.float32)

        # self.actionSize = 5  # Size of the robot's actions
        self.action_space = spaces.Discrete(5)

        self.targetDistance = 0  # Distance to target
        self.oldDistance = 0
        self.oldLinearVel = 0
        self.oldAngularVel = 0
        self.startDistance = 0
        self.reachTarget = 0.2  # Reached to target

        self.episodeN = 0
        self.stepN = 0
        self.success = False

        self.start_agentX = 0
        self.start_agentY = 0

        self.old_agentX = 0
        self.old_agentY = 0

        self.targetPointX = 0  # Target Pos X
        self.targetPointY = 0  # Target Pos Y

        # Means robot reached target point. True at beginning to calc random point in reset func
        self.isTargetReached = True
        self.goalCont = GoalController(self.map_name)
        self.agentController = AgentPosController(self.map_name)

        self.time_penalty = -0.05
        self.arrive_target = 200
        self.collided_penalty = -100
        self.minimum_step_reward = 0
        self.distancebounus = 10.0
        self.vel_diff_discount = 0.25
        self.Consecutive_NegRew = 0
        self.stuck = 30
        self.moving_dis_diff = []

    def pauseGazebo(self):
        '''
        Pause the simulation
        '''
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except Exception:
            print("/gazebo/pause_physics service call failed")

    def unpauseGazebo(self):
        '''
        Unpause the simulation
        '''
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except Exception:
            print("/gazebo/unpause_physics service call failed")

    def resetGazebo(self):
        '''
        Reset simualtion to initial phase
        '''
        rospy.logwarn("Start to reset Gazebo")
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except Exception:
            print("/gazebo/reset_simulation service call failed")
        rospy.logwarn("Success reset Gazebo")

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

    def getDepthImage(self):
        '''
        ROS callback function

        return depth image from the front camera
        '''
        while True:
            try:
                DepthImage = rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=5)
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

    def getOdomData(self):
        '''
        ROS callback function
        Modify odom data quaternion to euler

        return yaw, posX, posY of robot known as Pos2D
        '''
        while True:
            try:
                odom_raw_Data = rospy.wait_for_message('/odom', Odometry, timeout=5)
                odomData = odom_raw_Data.pose.pose
                velData = odom_raw_Data.twist.twist

                quat = odomData.orientation
                quatTuple = (
                    quat.x,
                    quat.y,
                    quat.z,
                    quat.w,
                )
                roll, pitch, yaw = tf.transformations.euler_from_quaternion(
                    quatTuple)
                robotX = odomData.position.x
                robotY = odomData.position.y

                robot_linearVel = math.sqrt(velData.linear.x ** 2 + velData.linear.y ** 2)
                robot_angularVel = velData.angular.z
                return yaw, robotX, robotY, robot_linearVel, robot_angularVel

            except Exception as e:
                rospy.logfatal("Error to get odom data " + str(e))

    def calcHeadingAngle(self, targetPointX, targetPointY, yaw, robotX, robotY, robot_linearVel, robot_angularVel):
        '''
        Calculate heading angle from robot to target

        return angle in float
        '''
        targetAngle = math.atan2(targetPointY - robotY, targetPointX - robotX)

        heading = targetAngle - yaw
        if heading > math.pi:
            heading -= 2 * math.pi

        elif heading < -math.pi:
            heading += 2 * math.pi

        return round(heading, 2)

    def calcDistance(self, x1, y1, x2, y2):
        '''
        Calculate euler distance of given two points

        return distance in float
        '''
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def calculateState(self, laserData, odomData):
        '''
        Modify laser data
        Calculate heading angle
        Calculate distance to target
        Calculate Linear velocity
        Calculate Angular velocity

        returns state as np.array

        State contains:
        laserData, heading, distance, obstacleMinRange, obstacleAngle
        '''

        heading = self.calcHeadingAngle(
            self.targetPointX, self.targetPointY, *odomData)
        _, robotX, robotY, robot_linearVel, robot_angularVel = odomData
        distance = self.calcDistance(
            robotX, robotY, self.targetPointX, self.targetPointY)

        isCrash = False  # If robot hit to an obstacle
        laserData = list(laserData.ranges)

        for i in range(len(laserData)):
            if i in self.front_side_lidar:
                if (self.minCrashRange > laserData[i] > 0):
                    isCrash = True
            else:
                if (self.minCrashRange_side > laserData[i] > 0):
                    isCrash = True
            if np.isinf(laserData[i]):
                laserData[i] = self.laserMaxRange
            if np.isnan(laserData[i]):
                laserData[i] = 0

        # obstacleMinRange = round(min(laserData), 2)
        # obstacleAngle = np.argmin(laserData)

        return laserData + [heading, distance, robot_linearVel, robot_angularVel], isCrash

    def _compute_reward(self, distanceToTarget, LinearVel, AngularVel, agentX, agentY, isCrash):

        reward = self.time_penalty
        done = False
        moving_dist = self.calcDistance(self.old_agentX, self.old_agentY, agentX, agentY)
        if len(self.moving_dis_diff) < self.stuck:
            self.moving_dis_diff.append(moving_dist)

        if distanceToTarget <= self.reachTarget:
            reward += self.arrive_target
            done = True
            rospy.logwarn("Reached to target!")
            return done, reward

        if isCrash:
            reward += self.collided_penalty
            done = True
            rospy.logwarn("Collision!")
            return done, reward

        if len(self.moving_dis_diff) == self.stuck:
            if sum(self.moving_dis_diff) <= 0.1:
                reward += self.collided_penalty
                done = True
                rospy.logwarn("Stuck!")
                return done, reward
            self.moving_dis_diff.pop(0)

        # 如過比上次更靠近目標就加分，反之就扣分，放大100倍
        reward_dist_to_target = (self.oldDistance - distanceToTarget) * self.distancebounus

        reward_Vel_diff = (math.fabs(self.oldLinearVel - LinearVel) + math.fabs(
            self.oldAngularVel - AngularVel)) * self.vel_diff_discount

        reward += reward_dist_to_target

        reward -= reward_Vel_diff

        # # 若該step的總reward小於-20分則結束
        # if reward <= self.minimum_step_reward:
        #     done = True

        self.oldDistance = distanceToTarget
        self.oldLinearVel = LinearVel
        self.oldAngularVel = AngularVel
        self.old_agentX = agentX
        self.old_agentY = agentY

        return done, reward

    def step(self, action):
        '''
        Act in envrionment
        After action return new state
        Calculate reward
        Calculate bot is crashed or not
        Calculate is episode done or not

        returns state as np.array

        State contains:
        laserData, heading, distance, obstacleMinRange, obstacleAngle, reward, done
        '''
        self.stepN += 1

        self.unpauseGazebo()

        """
        # Move
        maxAngularVel = 1.5
        angVel = ((self.actionSize - 1)/2 - action) * maxAngularVel / 2

        velCmd = Twist()
        velCmd.linear.x = 0.15
        velCmd.angular.z = angVel

        self.velPub.publish(velCmd)
        """

        # More basic actions
        if action == 0:
            velCmd = Twist()
            velCmd.linear.x = 0.0         #0.26
            velCmd.angular.z = 0.785      #-1.0
            self.velPub.publish(velCmd)
        elif action == 1:
            velCmd = Twist()
            velCmd.linear.x = 0.0         #0.26
            velCmd.angular.z = 0.262      #-0.5
            self.velPub.publish(velCmd)
        elif action == 2:
            velCmd = Twist()
            velCmd.linear.x = 0.26       #0.26
            velCmd.angular.z = 0.0       #0.0
            self.velPub.publish(velCmd)
        elif action == 3:
            velCmd = Twist()
            velCmd.linear.x = 0.0       #0.26
            velCmd.angular.z = -0.262   #0.5
            self.velPub.publish(velCmd)
        elif action == 4:
            velCmd = Twist()
            velCmd.linear.x = 0.0       #0.26
            velCmd.angular.z = -0.785    #1.0
            self.velPub.publish(velCmd)
        # elif action == 5:
        #     velCmd = Twist()
        #     velCmd.linear.x = 0.0       #0.13
        #     velCmd.angular.z = -0.524   #-1.0
        #     self.velPub.publish(velCmd)
        # elif action == 6:
        #     velCmd = Twist()
        #     velCmd.linear.x = 0.0       #0.13
        #     velCmd.angular.z = 0.785    #0.0
        #     self.velPub.publish(velCmd)
        # elif action == 7:
        #     velCmd = Twist()
        #     velCmd.linear.x = 0.0      #0.13
        #     velCmd.angular.z = -0.785  #1.0
        #     self.velPub.publish(velCmd)
        # elif action == 8:
        #     velCmd = Twist()
        #     velCmd.linear.x = 0.0
        #     velCmd.angular.z = -1.0
        #     self.velPub.publish(velCmd)
        # elif action == 9:
        #     velCmd = Twist()
        #     velCmd.linear.x = 0.0
        #     velCmd.angular.z = 0.0
        #     self.velPub.publish(velCmd)
        # elif action == 10:
        #     velCmd = Twist()
        #     velCmd.linear.x = 0.0
        #     velCmd.angular.z = 1.0
        #     self.velPub.publish(velCmd)

        # Observe
        laserData = self.getLaserData()
        odomData = self.getOdomData()
        depthImage = self.getDepthImage()

        _, robotX, robotY, robot_linearVel, robot_angularVel = odomData

        self.pauseGazebo()

        state, isCrash = self.calculateState(laserData, odomData)

        H, W = depthImage.shape
        self.state_image = np.reshape(depthImage, (H * W))
        self.state_sensor = np.asarray(state)

        observation = np.append(self.state_image, self.state_sensor)

        track = state[-4]
        distanceToTarget = state[-3]

        done, reward = self._compute_reward(distanceToTarget=distanceToTarget,
                                            LinearVel=robot_linearVel,
                                            AngularVel=robot_angularVel,
                                            agentX=robotX,
                                            agentY=robotY,
                                            isCrash=isCrash)

        # if reward < 0:
        #     self.Consecutive_NegRew += 1
        # else:
        #     self.Consecutive_NegRew = 0

        self.rewardSum += reward

        sys.stdout.write(
            "\r\x1b[K{}/{}==>reward: {:.5f}\t Total reward: {:.5f}\t track: {:.2f}\t action: {:.0f}\t now position = ({:.3f}, {:.3f})\t linear vel= {:.3f}\t angular vel= {:.3f}\t dist to target = {:.2f}({:.2f})\t collided = {}\n"
            .format(self.episodeN, self.stepN, reward, self.rewardSum, math.degrees(track), action, robotX, robotY,
                    robot_linearVel, robot_angularVel, distanceToTarget, self.startDistance, isCrash))

        sys.stdout.flush()

        # if self.rewardSum < -100 or self.Consecutive_NegRew == 40:
        #     done = True

        if done == True:
            finish_perct = (self.startDistance - distanceToTarget + self.reachTarget) / self.startDistance * 100
            if isCrash == True:
                print("Collide cause Done!!!")
            elif self.rewardSum < -100:
                print("Total reward too small  => cause Done!!!")
            elif distanceToTarget <= self.reachTarget:
                self.success = True
                finish_perct = 100.0
                print("Arrive => cause Done!!!")
            else:
                print("Step reward too small => cause Done!!!")

            # with open('DQN_model/validation.txt', 'a') as fp:
            #     fp.write(
            #         "Episode:{}({}),\t Take-off:({:.2f}, {:.2f}),\t Target:({:.2f}, {:.2f}),\t Dist:{:.2f},\t Success: {},\t Finish:{:.2f}%\n".format(
            #             self.episodeN, self.stepN, self.start_agentX, self.start_agentY, self.targetPointX,
            #             self.targetPointY, self.startDistance, self.success, finish_perct))

        info = {"x_pos": robotX, "y_pos": robotY}
        """
        done = False
        if isCrash:
            done = True

        distanceToTarget = state[-3]

        if distanceToTarget < 0.2:  # Reached to target
            self.isTargetReached = True

        if isCrash:
            reward = -150

        elif self.isTargetReached:
            # Reached to target
            rospy.logwarn("Reached to target!")
            reward = 200
            # Calc new target point
            self.targetPointX, self.targetPointY = self.goalCont.calcTargetPoint()
            self.isTargetReached = False

        else:
            # Neither reached to goal nor crashed calc reward for action
            yawReward = []
            currentDistance = state[-3]
            heading = state[-4]

            # Calc reward 
            # reference https://emanual.robotis.com/docs/en/platform/turtlebot3/ros2_machine_learning/

            for i in range(self.actionSize):
                angle = -math.pi / 4 + heading + (math.pi / 8 * i) + math.pi / 2
                tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
                yawReward.append(tr)

            try:
                distanceRate = 2 ** (currentDistance / self.targetDistance)
            except Exception:
                print("Overflow err CurrentDistance = ", currentDistance, " TargetDistance = ", self.targetDistance)
                distanceRate = 2 ** (currentDistance // self.targetDistance)

            reward = ((round(yawReward[action] * 5, 2)) * distanceRate)
        """

        return observation, reward, done, info

    def reset(self):
        '''
        Reset the envrionment
        Reset bot position

        returns state as np.array

        State contains:
        laserData, heading, distance, obstacleMinRange, obstacleAngle
        '''
        # Initialize the node
        # rospy.init_node('turtlebot3_gym_env', anonymous=True)

        # Connect to gazebo
        # self.velPub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.resetGazebo()

        self.targetPointX, self.targetPointY = self.goalCont.calcTargetPoint()

        rospy.logerr("Recalculating the target point!")

        while True:
            # Teleport bot to a random point
            self.start_agentX, self.start_agentY = self.agentController.teleportRandom()
            if self.calcDistance(self.targetPointX, self.targetPointY, self.start_agentX,
                                 self.start_agentY) > self.reachTarget * 10:
                break
            else:
                rospy.logerr("Reteleporting the bot!")
                time.sleep(2)
        self.old_agentX = self.start_agentX
        self.old_agentY = self.start_agentY
        self.moving_dis_diff = []
        """
        while True:
            # Teleport bot to a random point
            agentX, agentY = self.agentController.teleportRandom()
            if self.calcDistance(self.targetPointX, self.targetPointY, agentX, agentY) > self.minCrashRange:
                break
            else:
                rospy.logerr("Reteleporting the bot!")
                time.sleep(2)

        if self.isTargetReached:
            while True:
                self.targetPointX, self.targetPointY = self.goalCont.calcTargetPoint()
                if self.calcDistance(self.targetPointX, self.targetPointY, agentX, agentY) > self.minCrashRange:
                    self.isTargetReached = False
                    break
                else:
                    rospy.logerr("Recalculating the target point!")
                    time.sleep(2)
        """

        # Unpause simulation to make observation
        self.unpauseGazebo()
        laserData = self.getLaserData()
        odomData = self.getOdomData()
        depthImage = self.getDepthImage()
        self.pauseGazebo()

        state, isCrash = self.calculateState(laserData, odomData)

        H, W = depthImage.shape
        self.state_image = np.reshape(depthImage, (H * W))
        self.state_sensor = np.asarray(state)

        observation = np.append(self.state_image, self.state_sensor)

        self.targetDistance = state[-3]
        self.oldDistance = state[-3]
        self.startDistance = state[-3]
        # self.stateSize = len(state)

        self.stepN = 0
        self.episodeN += 1
        self.success = False
        self.rewardSum = 0
        self.Consecutive_NegRew = 0

        return observation  # Return state

    def render(self, mode='human'):
        return  # nothing

"""
There are 3 different maze map in this packet
After start one of them with launch file you have to edit this parameter
for that maze.

Options:
maze1
maze2
maze3
"""

def _random_points_within(poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds

    points = []

    while len(points) < num_points:
        random_point = pot([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            points.append(random_point)

    return points


class AgentPosController():
    '''
    This class control robot position
    We teleport our agent when environment reset
    So agent start from different position in every episode
    '''

    def __init__(self, map_name):
        self.agent_model_name = "turtlebot3_waffle"

        self.SELECT_MAP = map_name

        if self.SELECT_MAP == "maze1":
            # maze 1
            self.polygon_corner = [[3, 6.5], [3, -2], [0, -2], [0, -4], [-3.5, -6.5], [-3.5, -4], [-4.5, -4],
                                   [-4.5, -9],
                                   [-5.5, -9], [-5.5, -10], [-7.5, -10], [-7.5, -9], [-9, -9], [-9, -2.5], [-11, -2.5],
                                   [-12.5, -1], [-12.5, 1], [-10.67, 2.67], [-13, 5.33], [-10, 8], [-7, 5], [-4.5, 5],
                                   [-4.5, 2.5], [-3.5, 2.5], [-3.5, 6.5]]

            self.prohibited_area = [[[-4.33, 0.7], [-7, -2], [-5.7, -3], [-3.5, -0.33]],
                                    [[-6.5, 3.33], [-10.33, 7.33], [-12, 5.7], [-8, 1.5]],
                                    [[-9.5, 1], [-9.5, -1], [-12, -1], [-12, 1]],
                                    [[-5.5, -4], [-5.5, -8], [-8, -8], [-8, -4]]]

        elif self.SELECT_MAP == "maze2":
            # maze 2
            "With room"
            # self.polygon_corner = [[3.5, 0.5], [3.5, 5], [5, 5], [6.7, 6], [8.2, 5], [10.5, 5], [10.5, 10.7],[11.5, 12],
            #                        [14, 12], [15.5, 10.7], [15.5, 4], [17.2, 4], [27.2, -3.2], [25.5, -6.2], [22.9, -4.9],
            #                        [24.1, -2.9], [23.7, -2.6], [21.5, -5.5], [16.3, -2], [9.7, -2], [9.7, -3], [4, -3],
            #                        [4, 0], [5, 0], [5, 0.5]]

            "Without room"
            self.polygon_corner = [[3.5, 0.5], [3.5, 1.5], [10.5, 1.5], [10.5, 10.7], [11.5, 12],
                                   [14, 12], [15.5, 10.7], [15.5, 4], [17.2, 4], [27.2, -3.2], [25.5, -6.2],
                                   [22.9, -4.9],
                                   [24.1, -2.9], [23.7, -2.6], [21.5, -5.5], [16.3, -2], [16.3, 0.5]]

            "With room"
            # [[4.3, 3.8], [5, 4.6], [8.4, 4.6], [9, 3.8], [8.4, 2.8], [4.7, 2.8]],
            # [[9.8, 5], [9.8, 2], [3.5, 2], [3.5, 1.5], [10.5, 1.5], [10.5, 5]],
            # [[5.1, -0.8], [5.1, -2], [8.8, -2], [8.8, -0.8]],
            # [[5.5, 0.5], [5.5, 0], [9.7, 0], [9.7, -2], [11.1, -2], [11.1, 0], [14, 0],
            #  [14, -2], [16.3, -2], [16.3, 0.5]],
            self.prohibited_area = [
                [[11, 10], [14.6, 10], [14.6, 7.8], [11, 7.8]],
                [[11, 6.8], [11, 7.2], [14.2, 7.2], [14.2, 6.8]],
                [[15.5, 2.5], [15.5, 2], [16, 2], [16, 2.5]],
                [[20, 2.5], [19.5, 1], [22, -0.5], [23, 0.5]]
            ]

        elif self.SELECT_MAP == "maze3":
            self.polygon_corner = [[8.75, -5.0], [8.75, 2.75], [4.25, 2.75], [4.25, 5.0], [-1.75, 5.0], [-1.75, 2.75],
                                   [-8.75, 2.75], [-8.75, -5.0]]
            # self.polygon_corner = [[-2, 3.75], [1, 3.75], [1, 1.25], [-2, 1.25]]

            self.prohibited_area = [
                [[9.5, -5.5], [9.5, -1.5], [8.25, -1.5], [8.25, -4.5], [6.25, -4.5], [6.25, -5.5]],
                [[6.25, -1.75], [7.75, -1.75], [7.75, -4.25], [6.25, -4.25]],
                [[7.5, 0.25], [9.5, 0.25], [9.5, -2.25], [7.5, -2.25]],
                [[5.0, -0.5], [8, -0.5], [8, 2.5], [5.0, 2.5]],
                [[4.0, -3.5], [6.0, -3.5], [6.0, 0], [4.0, 0]],
                [[2.5, 0.5], [4.5, 0.5], [4.5, -3.5], [2.5, -3.5]],
                [[5.5, 1.25], [5.5, 5.0], [2.0, 5.0], [2.0, 4.0], [1.5, 4.0], [1.5, 1.25]],
                [[-2.5, 5.0], [-2.5, 3.25], [1.25, 3.25], [1.25, 5.0]],
                [[1.5, 0.0], [3.0, 0.0], [3.0, -1.5], [1.5, -1.5]],
                [[-1.5, 0.0], [-1.5, -3.75], [2.25, -3.75], [2.25, -2.75], [3.0, -2.75], [3.0, -0.75], [0.5, -0.75],
                 [0.5, 0.0]],
                [[5.25, -4.25], [5.25, -5.5], [-2.75, -5.5], [-2.75, -4.75], [-1.25, -4.75], [-1.25, -4.25]],
                [[-2.0, -5.5], [-2.0, -2.25], [-3.0, -2.25], [-3.0, -5.5]],
                [[-2.0, 3.5], [-2.0, 0.0], [-3.0, 0.0], [-3.0, 1.5], [-4.0, 1.5], [-4.0, 3.5]],
                [[-3.75, 3.25], [-3.75, 2.25], [-4.75, 2.25], [-4.75, 0.5], [-7.75, 0.5], [-7.75, 2.25], [-8.5, 2.25],
                 [-8.5, 3.25]],
                [[-8.0, 3.0], [-8.0, 1.25], [-9.75, 1.25], [-9.75, 3.0]],
                [[-9.25, 1.5], [-8.25, 1.5], [-8.25, 0.5], [-9.25, 0.5]],
                [[-8.75, -0.75], [-8.75, -2.0], [-3.25, -2.0], [-3.25, -0.75]],
                [[-5.75, -2.75], [-3.25, -2.75], [-3.25, -4.75], [-5.75, -4.75]],
                [[-7.75, -3.75], [-7.75, -5.0], [-6.25, -5.0], [-6.25, -3.75]],
                [[-9.0, -3.75], [-9.0, -5.25], [-7.75, -5.25], [-7.75, -3.75]],
                [[-3.75, 2.25], [-3.75, 0.25], [-2.25, 0.25], [-2.25, 2.25]],
                [[-8.0, -0.25], [-8.0, -1.75], [-4.5, -0.25], [-4.5, -1.75]],
                [[-3.25, -2.5], [-3.25, -4.25], [-2.0, -2.5], [-2.0, -4.25]],
                [[-1.5, 3.25], [1.0, 3.25], [1.0, 1.5], [-1.5, 1.5]],
                [[-5.25, -0.25], [-5.25, -1.75], [-3.75, -0.25], [-3.75, -1.75]]
            ]
        self.poly = Polygon(self.polygon_corner)

    def teleportRandom(self):
        '''

        Teleport agent return new x and y point

        return agent posX, posY in list
        '''

        model_state_msg = ModelState()
        model_state_msg.model_name = self.agent_model_name

        """Set the position of the begin"""
        while True:
            inside = False
            points = _random_points_within(self.poly, 1)
            xy_list = [[round(points[0].x, 2), round(points[0].y, 2)]]

            for i in range(len(self.prohibited_area)):
                path = mpltPath.Path(self.prohibited_area[i])
                inside = path.contains_points(xy_list)[0]
                if inside:
                    break
            if not inside:
                break
        # xy_list = [
        #     [-1.5,-1.5], [-0.5,-1.5], [-1.5,-0.5],
        #     [-0.5,1.5], [1.5,0.5],
        #     [2.5,2.5], [2.5,3.5], [1.5,3.5],
        # ]

        # Get random position for agent
        """
        # A representation of pose in free space, composed of position and orientation. 
        Point position
        Quaternion orientation
        """
        pose = Pose()
        pose.position.x, pose.position.y = random.choice(xy_list)

        model_state_msg.pose = pose
        """
        MSG: geometry_msgs/Point
        # This contains the position of a point in free space
        float64 x
        float64 y
        float64 z
        """
        model_state_msg.twist = Twist()

        """
        string reference_frame      
        # set pose/twist relative to the frame of this entity (Body/Model)
        # leave empty or "world" or "map" defaults to world-frame
        """
        model_state_msg.reference_frame = "world"

        # Start teleporting in Gazebo
        isTeleportSuccess = False
        for i in range(5):
            if not isTeleportSuccess:
                try:
                    rospy.wait_for_service('/gazebo/set_model_state')
                    telep_model_prox = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                    telep_model_prox(model_state_msg)
                    isTeleportSuccess = True
                    break
                except Exception as e:
                    rospy.logfatal("Error when teleporting agent " + str(e))
            else:
                rospy.logwarn("Trying to teleporting agent..." + str(i))
                time.sleep(2)

        if not isTeleportSuccess:
            rospy.logfatal("Error when teleporting agent")
            return "Err", "Err"

        return pose.position.x, pose.position.y


class GoalController():
    """
    This class controls target model and position
    """

    def __init__(self, map_name):
        self.model_path = "../models/gazebo/goal_sign/model.sdf"
        f = open(self.model_path, 'r')
        self.model = f.read()

        self.goal_position = Pose()
        self.goal_position.position.x = None  # Initial positions
        self.goal_position.position.y = None
        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        self.model_name = 'goal_sign'
        self.check_model = False  # This used to checking before spawn model if there is already a model

        self.SELECT_MAP = map_name

        if self.SELECT_MAP == "maze1":
            # maze 1
            self.polygon_corner = [[3, 6.5], [3, -2], [0, -2], [0, -4], [-3.5, -6.5], [-3.5, -4], [-4.5, -4],
                                   [-4.5, -9],
                                   [-5.5, -9], [-5.5, -10], [-7.5, -10], [-7.5, -9], [-9, -9], [-9, -2.5], [-11, -2.5],
                                   [-12.5, -1], [-12.5, 1], [-10.67, 2.67], [-13, 5.33], [-10, 8], [-7, 5], [-4.5, 5],
                                   [-4.5, 2.5], [-3.5, 2.5], [-3.5, 6.5]]

            self.prohibited_area = [[[-4.33, 0.7], [-7, -2], [-5.7, -3], [-3.5, -0.33]],
                                    [[-6.5, 3.33], [-10.33, 7.33], [-12, 5.7], [-8, 1.5]],
                                    [[-9.5, 1], [-9.5, -1], [-12, -1], [-12, 1]],
                                    [[-5.5, -4], [-5.5, -8], [-8, -8], [-8, -4]]]

        elif self.SELECT_MAP == "maze2":
            # maze 2
            "With room"
            # self.polygon_corner = [[3.5, 0.5], [3.5, 5], [5, 5], [6.7, 6], [8.2, 5], [10.5, 5], [10.5, 10.7],[11.5, 12],
            #                        [14, 12], [15.5, 10.7], [15.5, 4], [17.2, 4], [27.2, -3.2], [25.5, -6.2], [22.9, -4.9],
            #                        [24.1, -2.9], [23.7, -2.6], [21.5, -5.5], [16.3, -2], [9.7, -2], [9.7, -3], [4, -3],
            #                        [4, 0], [5, 0], [5, 0.5]]

            "Without room"
            self.polygon_corner = [[3.5, 0.5], [3.5, 1.5], [10.5, 1.5], [10.5, 10.7], [11.5, 12],
                                   [14, 12], [15.5, 10.7], [15.5, 4], [17.2, 4], [27.2, -3.2], [25.5, -6.2],
                                   [22.9, -4.9],
                                   [24.1, -2.9], [23.7, -2.6], [21.5, -5.5], [16.3, -2], [16.3, 0.5]]

            "With room"
            # [[4.3, 3.8], [5, 4.6], [8.4, 4.6], [9, 3.8], [8.4, 2.8], [4.7, 2.8]],
            # [[9.8, 5], [9.8, 2], [3.5, 2], [3.5, 1.5], [10.5, 1.5], [10.5, 5]],
            # [[5.1, -0.8], [5.1, -2], [8.8, -2], [8.8, -0.8]],
            # [[5.5, 0.5], [5.5, 0], [9.7, 0], [9.7, -2], [11.1, -2], [11.1, 0], [14, 0],
            #  [14, -2], [16.3, -2], [16.3, 0.5]],
            self.prohibited_area = [
                [[11, 10], [14.6, 10], [14.6, 7.8], [11, 7.8]],
                [[11, 6.8], [11, 7.2], [14.2, 7.2], [14.2, 6.8]],
                [[15.5, 2.5], [15.5, 2], [16, 2], [16, 2.5]],
                [[20, 2.5], [19.5, 1], [22, -0.5], [23, 0.5]]
            ]

        elif self.SELECT_MAP == "maze3":
            self.polygon_corner = [[8.75, -5.0], [8.75, 2.75], [4.25, 2.75], [4.25, 5.0], [-1.75, 5.0], [-1.75, 2.75],
                                   [-8.75, 2.75], [-8.75, -5.0]]
            # self.polygon_corner = [[-2, 3.75], [1, 3.75], [1, 1.25], [-2, 1.25]]

            self.prohibited_area = [
                [[9.5, -5.5], [9.5, -1.5], [8.25, -1.5], [8.25, -4.5], [6.25, -4.5], [6.25, -5.5]],
                [[6.25, -1.75], [7.75, -1.75], [7.75, -4.25], [6.25, -4.25]],
                [[7.5, 0.25], [9.5, 0.25], [9.5, -2.25], [7.5, -2.25]],
                [[5.0, -0.5], [8, -0.5], [8, 2.5], [5.0, 2.5]],
                [[4.0, -3.5], [6.0, -3.5], [6.0, 0], [4.0, 0]],
                [[2.5, 0.5], [4.5, 0.5], [4.5, -3.5], [2.5, -3.5]],
                [[5.5, 1.25], [5.5, 5.0], [2.0, 5.0], [2.0, 4.0], [1.5, 4.0], [1.5, 1.25]],
                [[-2.5, 5.0], [-2.5, 3.25], [1.25, 3.25], [1.25, 5.0]],
                [[1.5, 0.0], [3.0, 0.0], [3.0, -1.5], [1.5, -1.5]],
                [[-1.5, 0.0], [-1.5, -3.75], [2.25, -3.75], [2.25, -2.75], [3.0, -2.75], [3.0, -0.75], [0.5, -0.75],
                 [0.5, 0.0]],
                [[5.25, -4.25], [5.25, -5.5], [-2.75, -5.5], [-2.75, -4.75], [-1.25, -4.75], [-1.25, -4.25]],
                [[-2.0, -5.5], [-2.0, -2.25], [-3.0, -2.25], [-3.0, -5.5]],
                [[-2.0, 3.5], [-2.0, 0.0], [-3.0, 0.0], [-3.0, 1.5], [-4.0, 1.5], [-4.0, 3.5]],
                [[-3.75, 3.25], [-3.75, 2.25], [-4.75, 2.25], [-4.75, 0.5], [-7.75, 0.5], [-7.75, 2.25], [-8.5, 2.25],
                 [-8.5, 3.25]],
                [[-8.0, 3.0], [-8.0, 1.25], [-9.75, 1.25], [-9.75, 3.0]],
                [[-9.25, 1.5], [-8.25, 1.5], [-8.25, 0.5], [-9.25, 0.5]],
                [[-8.75, -0.75], [-8.75, -2.0], [-3.25, -2.0], [-3.25, -0.75]],
                [[-5.75, -2.75], [-3.25, -2.75], [-3.25, -4.75], [-5.75, -4.75]],
                [[-7.75, -3.75], [-7.75, -5.0], [-6.25, -5.0], [-6.25, -3.75]],
                [[-9.0, -3.75], [-9.0, -5.25], [-7.75, -5.25], [-7.75, -3.75]],
                [[-3.75, 2.25], [-3.75, 0.25], [-2.25, 0.25], [-2.25, 2.25]],
                [[-8.0, -0.25], [-8.0, -1.75], [-4.5, -0.25], [-4.5, -1.75]],
                [[-3.25, -2.5], [-3.25, -4.25], [-2.0, -2.5], [-2.0, -4.25]],
                [[-1.5, 3.25], [1.0, 3.25], [1.0, 1.5], [-1.5, 1.5]],
                [[-5.25, -0.25], [-5.25, -1.75], [-3.75, -0.25], [-3.75, -1.75]]
            ]

        self.poly = Polygon(self.polygon_corner)

    def respawnModel(self):
        '''
        Spawn model in Gazebo
        '''
        rospy.logwarn("Start to spawn target model!")
        isSpawnSuccess = False
        for i in range(5):
            if not self.check_model:  # This used to checking before spawn model if there is already a model
                try:
                    rospy.wait_for_service('gazebo/spawn_sdf_model')
                    spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                    spawn_model_prox(self.model_name, self.model, 'robotos_name_space', self.goal_position, "world")
                    isSpawnSuccess = True
                    self.check_model = True
                    break
                except Exception as e:
                    rospy.logfatal("Error when spawning the goal sign " + str(e))
            else:
                rospy.logwarn("Trying to spawn goal sign ..." + str(i))
                time.sleep(2)

        if not isSpawnSuccess:
            rospy.logfatal("Error when spawning the goal sign")

        rospy.logwarn("Success spawn target model!")

    def deleteModel(self):
        '''
        Delete model from Gazebo
        '''
        rospy.logwarn("Start to delete target model!")

        while True:
            if self.check_model:
                try:
                    rospy.wait_for_service('gazebo/delete_model')
                    del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                    del_model_prox(self.model_name)

                    rospy.wait_for_service('gazebo/get_model_state')
                    check_model_prox = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
                    goal_sign_state = GetModelStateRequest()
                    goal_sign_state.model_name = self.model_name
                    goal_sign_state.relative_entity_name = "world"
                    obj_state = check_model_prox(goal_sign_state)
                    if obj_state.status_message == 'GetModelState: model does not exist':
                        self.check_model = False
                        break
                except Exception as e:
                    rospy.logfatal("Error when deleting the goal sign " + str(e))
            else:
                break

    def calcTargetPoint(self):
        """
        This function return a target point randomly for robot
        """
        self.deleteModel()
        # Wait for deleting
        time.sleep(0.5)
        rospy.logwarn("Success delete target model!")

        while True:
            inside = False
            points = _random_points_within(self.poly, 1)
            goal_xy_list = [[round(points[0].x, 2), round(points[0].y, 2)]]

            for i in range(len(self.prohibited_area)):
                path = mpltPath.Path(self.prohibited_area[i])
                inside = path.contains_points(goal_xy_list)[0]
                if inside:
                    break
            if not inside:
                break

        # goal_xy_list = [
        #     [-1.5, 0.5], [-1.5, 1.5], [-0.5, 0.5], [-0.5, 1.5],
        #     [0.5, -0.5], [0.5, -1.5], [2.5, -0.5], [2.5, 0.5],
        #     [5.5,-1.5], [5.5,-0.5], [5.5,0.5], [5.5,1.5]
        # ]

        self.goal_position.position.x, self.goal_position.position.y = random.choice(goal_xy_list)
        rospy.logwarn("Success caculate target point!")
        # Check last goal position not same with new goal
        # while True:
        #     self.goal_position.position.x, self.goal_position.position.y = random.choice(goal_xy_list)
        #
        #     if self.last_goal_x != self.goal_position.position.x:
        #         if self.last_goal_y != self.goal_position.position.y:
        #             break

        # Spawn goal model
        self.respawnModel()

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        # Inform user
        rospy.logwarn(
            "New goal position : " + str(self.goal_position.position.x) + " , " + str(self.goal_position.position.y))

        return self.goal_position.position.x, self.goal_position.position.y

    def getTargetPoint(self):
        return self.goal_position.position.x, self.goal_position.position.y


