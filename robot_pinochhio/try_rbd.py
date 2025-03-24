import rclpy
import numpy as np
from numpy.linalg import norm, solve
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String
from xml.etree.ElementTree import XML, fromstring
import pinocchio as pin


ENDEF_FRAME_ID = 87

class PinocchioTest(Node):

    def __init__(self):
        super().__init__('pinocchio_test_node')
        self.subscription = self.create_subscription(String,'/robot_description',self.listener_callback,QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL,depth=1))
        self.subscription  # prevent unused variable warning

        self.model = None
        self.data = None
        self.geom_model = None
        self.geom_data = None
        np.set_printoptions(suppress=True)


        # Uncomment to make calculations independent of data retrieval
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def listener_callback(self, msg):

        # TASK 1
        self.extractModelDataGeomGeomData(msg)
        # self.model.saveToText("node_parsed.txt")

    def timer_callback(self):

        q = self.getQ()
        self.calculate_fk_jacobian(q)
        
        # TASK 2
        # self.log_transforms()

        # TASK 3
        #endeff_j = pin.getFrameJacobian(self.model, self.data, ENDEF_FRAME_ID, pin.WORLD)
        # self.get_logger().info(f"\n Frame of endeffector \n {self.data.oMf[ENDEF_FRAME_ID]}")
        # self.get_logger().info(f"\n Jacobian of endeffector \n {np.round(endeff_j,decimals = 2)}")
        # self.get_logger().info(f"\n Jocabian of arm tool frame \n {np.round(pin.computeFrameJacobian(model, self.data, q, 73, pin.WORLD), decimals=2)}")

        # TASK 4
        # # Add collisition pairs
        # self.geom_model.addAllCollisionPairs()
        # self.get_logger().info(f"num collision pairs - initial: {len(self.geom_model.collisionPairs)}")
        #
        # self.generate_data()
        #
        # # Compute all the collisions
        # pin.computeCollisions(self.model, self.data, self.geom_model, self.geom_data, q, False)
        #
        # # Print the status of collision for all collision pairs
        # self.log_nos_collision()
        # self.log_collisions()
        #
        # # Compute for a single pair of collision
        # pin.updateGeometryPlacements(self.model, self.data, self.geom_model, self.geom_data, q)
        # pin.computeCollision(self.geom_model, self.geom_data, 0)
        # cr = self.geom_data.collisionResults[0]
        # self.get_logger().info(f"Collision for collision pair 0 is: {cr.isCollision()}")


        # MAXIMUM PAYLOAD FOR GIVEN CONFIGURATION

        payload, tau = self.calculate_max_payload(q)

        # TORQUE LIMITS REQUIRED FOR GIVEN PAYLOAD

        tau = self.tau_qPayload(payload)
        self.log_tau(tau)



    # DATA HANDLING METHODS
    def calculate_fk_jacobian(self,q):
        # Perform the forward kinematics over the kinematic tree
        pin.forwardKinematics(self.model, self.data, q)   
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def calculate_max_payload(self,q,F_est = 500,F_err = 1,log = True):

        if self.feasible_payload(q,F_est):
            while self.feasible_payload(q,F_est):
                F_est = 2*F_est
            lb = F_est/2
            ub = F_est
        else:
            while not self.feasible_payload(q,F_est):
                F_est = F_est/2
                if F_est<F_err:
                    return 0, self.grav_tau(q)
            lb = F_est
            ub = F_est*2

        while (ub-lb)>F_err:
            mid = (lb+ub)/2
            if self.feasible_payload(q,mid):
                lb = mid
            else:
                ub = mid
        f_max = (lb+ub)/2
        tau = self.tau_qPayload(q,f_max)

        if log:
            self.get_logger().info(f"Max payload for given configuration is {f_max}. \n Torques required for these are: value [max]")
            for i in range(self.model.nv):
                self.get_logger().info(f"Joint with Joint ID {i} with torque {np.round(tau[i],decimals=2)} [{self.model.effortLimit[i]}]")

        return f_max, tau

    def getQ(self, q_in = None, zeros = False, log = True):

        if q_in is not None:
            q = q_in
        elif zeros:
            q = np.zeros(self.model.nq)
        else:
            # Sample a random configuration
            # q = pin.randomConfiguration(self.model)
            q = np.random.rand(self.model.nq)*6.28-3.14

        if log:
            self.get_logger().info(f"Using configuration q:\n {q}")

        return q

    def tau_qFext(self,q,fext):
        pin.rnea(self.model, self.data, q, np.zeros(self.model.nv), np.zeros(self.model.nv),fext)
        return self.data.tau
    
    def grav_tau(self,q):
        f = [pin.Force(np.zeros(6)) for _ in range(self.model.njoints)]
        return self.tau_qFext(q,f) 
    
    def get_null_force(self):
        return [pin.Force(np.zeros(6)) for _ in range(self.model.njoints)]

    def payload_joint_force(self,payload):
        f = self.get_null_force()
        f[self.model.frames[ENDEF_FRAME_ID].parentJoint] = pin.Force(np.array([0.0,0.0,-payload,0.0,0.0,0.0]))
        return f
    
    def tau_qPayload(self,q,payload):
        f = self.payload_joint_force(payload)
        return self.tau_qFext(q,f)
        
    def tau_within_limit(self, tau):
        limits = self.model.effortLimit
        out = True
        for i in range(self.model.nv):
            if abs(tau[i])>limits[i]:
                out = False
                break
        return out

    def feasible_payload(self, q, payload):
        tau = self.tau_qPayload(q,payload)
        return self.tau_within_limit(tau)
    
    # DATA GENERATION METHODS
    def extractModelDataGeomGeomData(self, msg):

        model = pin.buildModelFromXML(msg.data)

        if model is not None:
            if model != self.model:
                self.get_logger().info("Successfully retrieved model with name: " + model.name)
                self.model = model
        else:
            self.get_logger().info("Failed to generate model")

        self.geom_model = pin.buildGeomFromUrdfString(model, msg.data, pin.GeometryType.COLLISION)

        self.get_logger().info(f"Found {self.geom_model.ngeoms} geometry objects")

        # Create data required by the algorithms
        self.generate_data()

    def generate_data(self):
        self.data = self.model.createData()
        self.geom_data = pin.GeometryData(self.geom_model)

    #LOGGING METHODS
    def log_joint_names(self):
        for i in range(len(self.model.names)):
            self.get_logger().info(f"\n Joint with joint id {i} is {self.model.names[i]}")

    def log_transforms(self):
        for oMi, name in zip(self.data.oMi,self.model.names):
            self.get_logger().info(f"\n Transform for joint {name} is: \n {oMi.homogeneous}")

    def log_frame_names(self):
        for i in range(len(self.model.frames)):
            self.get_logger().info(f"\n Frame with frame id {i} is {self.model.frames[i].name}")

    def log_nos_collision(self):
        n = 0
        for k in range(len(self.geom_model.collisionPairs)):
            cr = self.geom_data.collisionResults[k]
            if cr.isCollision():
                n = n+1
        self.get_logger().info(f"Total number of collisions found: {n}")

    def log_collisions(self):
        for k in range(len(self.geom_model.collisionPairs)):
            cr = self.geom_data.collisionResults[k]
            cp = self.geom_model.collisionPairs[k]
            if cr.isCollision():
                self.get_logger().info(f"collision pair: {cp.first} , {cp.second} , - collision: {cr.isCollision()}")

    def log_joint_effort_limits(self):
        for i in range(self.model.nv):
            self.get_logger().info(f"Joint with Joint ID {i} with limiting torque {self.model.effortLimit[i]}")

    def log_tau(self,tau):
        for i in range(self.model.nv):
            if tau[i]!=0:
                self.get_logger().info(f"Joint with Joint ID {i} with torque {tau[i]}")

    def log_force(self):
        for i in range(self.model.njoints):
            if (self.data.of[i].linear != self.get_null_force()[i].linear).any or (self.data.of[i].angular != self.get_null_force()[i].angular).any:
                self.get_logger().info(f"Joint with Joint ID {i} with force {self.data.of[i]}")


def main(args=None):
    rclpy.init(args=args)

    pin_test = PinocchioTest()

    rclpy.spin(pin_test)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pin_test.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
























"""
INFORMATION OF THE ROBOT 

LINKS

FRAME ID  PARENT FRAME ID PARENT JOINT ID FRAME NAME
0   0   0   universe
1   0   0   base_footprint
2   1   0   base_footprint_joint
3   2   0   base_link
4   3   0   base_antenna_left_joint
5   4   0   base_antenna_left_link
6   3   0   base_antenna_right_joint
7   6   0   base_antenna_right_link
8   3   0   base_dock
9   8   0   base_dock_link
10   3   0   base_imu_joint
11   10   0   base_imu_link
12   3   0   base_laser_joint
13   12   0   base_laser_link
14   3   0   base_mic_back_left_joint
15   14   0   base_mic_back_left_link
16   3   0   base_mic_back_right_joint
17   16   0   base_mic_back_right_link
18   3   0   base_mic_front_left_joint
19   18   0   base_mic_front_left_link
20   3   0   base_mic_front_right_joint
21   20   0   base_mic_front_right_link
22   3   0   base_sonar_01_joint
23   22   0   base_sonar_01_link
24   3   0   base_sonar_02_joint
25   24   0   base_sonar_02_link
26   3   0   base_sonar_03_joint
27   26   0   base_sonar_03_link
28   3   0   caster_back_left_1_joint
29   28   0   caster_back_left_1_link
30   29   0   caster_back_left_2_joint
31   30   0   caster_back_left_2_link
32   3   0   caster_back_right_1_joint
33   32   0   caster_back_right_1_link
34   33   0   caster_back_right_2_joint
35   34   0   caster_back_right_2_link
36   3   0   caster_front_left_1_joint
37   36   0   caster_front_left_1_link
38   37   0   caster_front_left_2_joint
39   38   0   caster_front_left_2_link
40   3   0   caster_front_right_1_joint
41   40   0   caster_front_right_1_link
42   41   0   caster_front_right_2_joint
43   42   0   caster_front_right_2_link
44   3   0   suspension_left_joint
45   44   0   suspension_left_link
46   45   1   wheel_left_joint
47   46   1   wheel_left_link
48   3   0   suspension_right_joint
49   48   0   suspension_right_link
50   49   2   wheel_right_joint
51   50   2   wheel_right_link
52   3   0   torso_fixed_column_joint
53   52   0   torso_fixed_column_link
54   3   0   torso_fixed_joint
55   54   0   torso_fixed_link
56   55   3   torso_lift_joint
57   56   3   torso_lift_link
58   57   4   arm_1_joint
59   58   4   arm_1_link
60   59   5   arm_2_joint
61   60   5   arm_2_link
62   61   6   arm_3_joint
63   62   6   arm_3_link
64   63   7   arm_4_joint
65   64   7   arm_4_link
66   65   8   arm_5_joint
67   66   8   arm_5_link
68   67   9   arm_6_joint
69   68   9   arm_6_link
70   69   10   arm_7_joint
71   70   10   arm_7_link
72   71   10   arm_tool_joint
73   72   10   arm_tool_link
74   73   10   wrist_ft_joint
75   74   10   wrist_ft_link
76   75   10   wrist_tool_joint
77   76   10   wrist_ft_tool_link
78   77   10   gripper_joint
79   78   10   gripper_link
80   79   10   gripper_grasping_frame_joint
81   80   10   gripper_grasping_frame
82   79   11   gripper_left_finger_joint
83   82   11   gripper_left_finger_link
84   79   12   gripper_right_finger_joint
85   84   12   gripper_right_finger_link
86   77   10   gripper_tool_joint
87   86   10   gripper_tool_link        # end effector
88   57   13   head_1_joint
89   88   13   head_1_link
90   89   14   head_2_joint
91   90   14   head_2_link
92   91   14   head_front_camera_joint
93   92   14   head_front_camera_link
94   93   14   head_front_camera_optical_joint
95   94   14   head_front_camera_optical_frame
96   93   14   head_front_camera_orbbec_aux_joint
97   96   14   head_front_camera_orbbec_aux_joint_frame
98   97   14   head_front_camera_rgb_joint
99   98   14   head_front_camera_rgb_frame
100   99   14   head_front_camera_depth_joint
101   100   14   head_front_camera_depth_frame
102   101   14   head_front_camera_depth_optical_joint
103   102   14   head_front_camera_depth_optical_frame
104   99   14   head_front_camera_rgb_optical_joint
105   104   14   head_front_camera_rgb_optical_frame
106   1   0   cover_joint
107   106   0   base_cover_link



JOINTS
universe
wheel_left_joint
wheel_right_joint
torso_lift_joint
arm_1_joint
arm_2_joint
arm_3_joint
arm_4_joint
arm_5_joint
arm_6_joint
arm_7_joint
gripper_left_finger_joint
gripper_right_finger_joint
head_1_joint
head_2_joint



"""