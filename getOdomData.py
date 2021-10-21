import rospy
import math
import tf
import geometry_msgs.msg
import turtlesim.srv
import threading
import time

def getOdomData():
    global robotX
    global robotY
    global roll
    global pitch
    global yaw

    listener = tf.TransformListener()
    rate = rospy.Rate(10.0)

    while not rospy.is_shutdown():
        try:
            (trans,rot) = listener.lookupTransform('/map', '/laser', rospy.Time(0))
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

if __name__ == '__main__':
    rospy.init_node('turtle_tf_listener')

    t_getOdomData = threading.Thread(target=getOdomData)

    t_getOdomData.start()

    start = time.time()

    time.sleep(5)

    while 1:
        print("X:{}, Y:{}, roll:{}, pitch:{}, yaw:{}, time:{}".format(robotX, robotY, roll, pitch, yaw,
                                                                      time.time() - start))

