import numpy as np
import pinocchio as pin
import os
from pathlib import Path
from pinocchio.visualize import RVizVisualizer



ENDEF_FRAME_ID = 87

tiago_robot_dir = "/home/kushal/colcon_ws/src/tiago_robot"
tiago_desc_dir = tiago_robot_dir + "/tiago_description"
xacro_dir = tiago_desc_dir + "/robots"
xacro_file = xacro_dir + "/tiago.urdf.xacro"
meshes_dir = tiago_desc_dir + "/meshes"
urdf_file = Path(__file__).parent.parent/"resource/tiago.urdf"
 

class PinocchioTest():

    def __init__(self):
        
        self.model = None
        self.data = None
        self.geom_model = None
        self.geom_data = None
        np.set_printoptions(suppress=True)
        

    def functionality(self):

        # TASK 1
        # self.extractModelDataGeomGeomData(msg)
        self.fromFile()




        q = self.getQ()

        # viz = RVizVisualizer(self.model)

        # # Initialize the viewer.
        # viz.initViewer()
        # viz.loadViewerModel("pinocchio")
        # viz.display(q)

        self.calculate_fk_jacobian(q)
        
        # TASK 2
        # self.log_transforms()

        # TASK 3
        endeff_j = pin.getFrameJacobian(self.model, self.data, ENDEF_FRAME_ID, pin.WORLD)
        # print(f"\n Frame of endeffector \n {self.data.oMf[ENDEF_FRAME_ID]}")
        # print(f"\n Jacobian of endeffector \n {endeff_j}")
        # print(f"\n Jocabian of arm tool frame \n {pin.computeFrameJacobian(model, self.data, q, 73, pin.WORLD)}")


        # TASK 4
        # Add collisition pairs
        self.geom_model.addAllCollisionPairs()
        # print(f"num collision pairs - initial: {len(self.geom_model.collisionPairs)}")
        
        self.generate_data()

        # Compute all the collisions
        pin.computeCollisions(self.model, self.data, self.geom_model, self.geom_data, q, False)
        
        # Print the status of collision for all collision pairs
        # self.log_no_collision()
        # self.log_collisions()
        
        # Compute for a single pair of collision
        pin.updateGeometryPlacements(self.model, self.data, self.geom_model, self.geom_data, q)
        pin.computeCollision(self.geom_model, self.geom_data, 0)
        cr = self.geom_data.collisionResults[0]
        # print(f"Collision for collision pair 0 is: {cr.isCollision()}")
        # print(self.model.njoints)
        # print(self.model.names[20])



        # self.log_joint_effort_limits()

        # CALCULATING JOINT TORQUES FOR A GIVEN CONFIGURATION AND Fext

        # When fext = 0, joint torque is purely due to gravity

        zero_payload_tau = self.grav_tau(q)
                # This is equivalent to:
                # f = self.get_null_force()
                # zero_payload_tau = self.static_payload_torque(q,f) # also can be called gravity torque 

        

        f = self.get_null_force()
        f[self.model.frames[ENDEF_FRAME_ID].parentJoint] = pin.Force(np.array([0.0,0.0,-100,0.0,0.0,0.0]))

        payload_torque = self.static_payload_torque(q,f)

        print("Gravity torque:")
        self.log_tau(zero_payload_tau)

        print("Payload Torque for given load")
        self.log_force()
        self.log_tau(payload_torque)


        # ACCOUNTING EFFORT LIMITS

        # self.log_joint_effort_limits()
        

    def extractModelDataGeomGeomData(self, msg):

        model = pin.buildModelFromXML(msg.data)

        if model is not None:
            print("Successfully retrieved model with name: " + model.name)
            self.model = model
        else:
            print("Failed to generate model")

        self.geom_model = pin.buildGeomFromUrdfString(model, msg.data, pin.GeometryType.COLLISION)

        # Create data required by the algorithms
        self.data = self.model.createData()
        self.geom_data = pin.GeometryData(self.geom_model)
    
    def fromFile(self):
        os.system(f'xacro {xacro_file} > {urdf_file}')
        self.model = pin.buildModelFromUrdf(urdf_file)
        self.geom_model = pin.buildGeomFromUrdf(self.model, urdf_file, pin.GeometryType.COLLISION)
        self.generate_data()

    def generate_data(self):
        self.data = self.model.createData()
        self.geom_data = pin.GeometryData(self.geom_model)
    
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
            print(f"Using configuration q: {q.T}")

        return q

    def log_joint_names(self):
        for i in range(len(self.model.names)):
            print(f"\n Joint with joint id {i} is {self.model.names[i]}")

    def log_transforms(self):
        for oMi, name in zip(self.data.oMi,self.model.names):
            print(f"\n Transform for joint {name} is: \n {oMi.homogeneous}")

    def log_frame_names(self):
        for i in range(len(self.model.frames)):
            print(f"\n Frame with frame id {i} is {self.model.frames[i].name}")

    def calculate_fk_jacobian(self,q):
        # Perform the forward kinematics over the kinematic tree
        pin.forwardKinematics(self.model, self.data, q)   
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
    
    def log_no_collision(self):
        n = 0
        for k in range(len(self.geom_model.collisionPairs)):
            cr = self.geom_data.collisionResults[k]
            if cr.isCollision():
                n = n+1
        print(f"Total number of collisions found: {n}")

    def log_collisions(self):
        for k in range(len(self.geom_model.collisionPairs)):
            cr = self.geom_data.collisionResults[k]
            cp = self.geom_model.collisionPairs[k]
            if cr.isCollision():
                print(f"collision pair: {cp.first} , {cp.second} , - collision: {cr.isCollision()}")

    def log_joint_effort_limits(self):
        print(self.model.effortLimit)

    def log_tau(self,tau):
        for i in range(self.model.nv):
            if tau[i]!=0:
                print(f"Joint {i} with torque {tau[i]}")

    def log_force(self):
        for i in range(self.model.njoints):
            if (self.data.of[i].linear != self.get_null_force()[i].linear).any or (self.data.of[i].angular != self.get_null_force()[i].angular).any:
                print(f"Joint {i} with force {self.data.of[i]}")

    def static_payload_torque(self,q,f):
        pin.rnea(self.model, self.data, q, np.zeros(self.model.nv), np.zeros(self.model.nv),f)
        return self.data.tau
    
    def grav_tau(self,q):
        f = [pin.Force(np.zeros(6)) for _ in range(self.model.njoints)]
        return self.static_payload_torque(q,f) 
    
    def get_null_force(self):
        return [pin.Force(np.zeros(6)) for _ in range(self.model.njoints)]



        


def main(args=None):

    pin_test = PinocchioTest()
    pin_test.functionality()



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