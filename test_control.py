import rospy
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

def unpauseGazebo():
    '''
    Unpause the simulation
    '''
    rospy.wait_for_service('/gazebo/unpause_physics')
    try:
        unpause
    except Exception:
        print("/gazebo/unpause_physics service call failed")

def pauseGazebo():
    '''
    Pause the simulation
    '''
    rospy.wait_for_service('/gazebo/pause_physics')
    try:
        pause
    except Exception:
        print("/gazebo/pause_physics service call failed")

def resetGazebo():
    '''
    Reset simualtion to initial phase
    '''
    rospy.logwarn("Start to reset Gazebo")
    rospy.wait_for_service('/gazebo/reset_simulation')
    try:
        reset_proxy
    except Exception:
        print("/gazebo/reset_simulation service call failed")
    rospy.logwarn("Success reset Gazebo")

if __name__ == '__main__':

    rospy.init_node('turtlebot3_gym_env', anonymous=True)
    velPub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
    unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    reset_proxy = rospy.ServiceProxy(
        '/gazebo/reset_simulation', Empty)

    print("Start!!!!")
    while 1:
        print("step")
        unpauseGazebo()
        velCmd = Twist()
        velCmd.linear.x = 0.26
        velCmd.angular.z = -1.0
        velPub.publish(velCmd)
        pauseGazebo()

        print("reset")
        resetGazebo()
        # rospy.init_node('turtlebot3_gym_env', anonymous=True)
        # velPub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)