from mavros_msg.srv import CommandLong
import rospy

rospy.wait_for_service('/mavros/cmd/command')
cmd = rospy.ServiceProxy('mavros/cmd/commad', CommandLong)

cmd(
    command = 211, # MAV_CMD_DO_GRIPPER)
    param1=1, # gripper ID
    param2=1, # action (1=close, 0=open)
    param3=0,
    param4=0,
    param5=0,
    param6=0,
    param7=0,
    Confirmation=0
)
