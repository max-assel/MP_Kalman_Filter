#!/usr/bin/env python
import getch
import rospy
from std_msgs.msg import String #String message
from geometry_msgs.msg import Point, Twist

lin_vel  = 0
ang_vel = 0
lin_factor = 0.25
ang_factor = 0.25
################################
# created by yuvaram
#yuvaramsingh94@gmail.com
################################
pub = rospy.Publisher('tb3_0/cmd_vel', Twist, queue_size=5)

def keys():
    global lin_vel, ang_vel
    rate = rospy.Rate(10)#try removing this line and see what happens
    while not rospy.is_shutdown():
        key = getch.getch()
        print('key: ', key)
        twist = Twist()
        if key == 'w':
            lin_vel += 1.0 * lin_factor
        elif key == 'x':
            lin_vel -= 1.0 * lin_factor
        elif key == 'a':
            ang_vel += 1.0 * ang_factor
        elif key == 'd':
            ang_vel -= 1.0 * ang_factor
        elif key == 's':
            lin_vel = 0
            ang_vel = 0
        elif key == '\x03':
            break
        twist.linear.x = lin_vel
        twist.angular.z = ang_vel
        print('current velocity: ', twist)
        pub.publish(twist)



if __name__=='__main__':
    try:
        rospy.init_node('keyboard_teleop', anonymous=True)
        keys()
    except rospy.ROSInterruptException:
        pass