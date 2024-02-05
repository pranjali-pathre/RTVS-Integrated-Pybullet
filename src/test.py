# import pybullet as p
# import time
# import pybullet_data

# p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.loadURDF("table/table.urdf", 0.5000000, 0.00000, -.820000, 0.000000, 0.000000, 0.0, 1.0)
# p.setGravity(0, 0, -10)

# while (1):
#   p.stepSimulation()
#   time.sleep(10000)
#   #p.saveWorld("test.py")
#   viewMat = p.getDebugVisualizerCamera()[2]
#   #projMatrix = [0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
#   projMatrix = p.getDebugVisualizerCamera()[3]
#   width = 640
#   height = 480
#   img_arr = p.getCameraImage(width=width,
#                              height=height,
#                              viewMatrix=viewMat,
#                              projectionMatrix=projMatrix)

import pybullet as p
import pybullet_data

p.connect(p.SHARED_MEMORY)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

pr2_gripper = 2
pr2_cid = 1

CONTROLLER_ID = 0
POSITION = 1
ORIENTATION = 2
ANALOG = 3
BUTTONS = 6

gripper_max_joint = 0.550569
while True:
  events = p.getVREvents()
  for e in (events):
    if e[CONTROLLER_ID] == 3:  # To make sure we only get the value for one of the remotes
      p.changeConstraint(pr2_cid, e[POSITION], e[ORIENTATION], maxForce=500)
      p.setJointMotorControl2(pr2_gripper,
                              0,
                              controlMode=p.POSITION_CONTROL,
                              targetPosition=gripper_max_joint - e[ANALOG] * gripper_max_joint,
                              force=1.0)
      p.setJointMotorControl2(pr2_gripper,
                              2,
                              controlMode=p.POSITION_CONTROL,
                              targetPosition=gripper_max_joint - e[ANALOG] * gripper_max_joint,
                              force=1.1)
      

      # fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100
    # projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)
    # com_p, com_o, _, _, _, _ = p.getLinkState(robot, 7, computeForwardKinematics=True)
    # rot_matrix = p.getMatrixFromQuaternion(com_o)
    # rot_matrix = np.array(rot_matrix).reshape(3, 3)
    # # Initial vectors
    # init_camera_vector = (0, 0, 1) # z-axis
    # init_up_vector = (0, 1, 0) # y-axis
    # # Rotated vectors
    # camera_vector = rot_matrix.dot(init_camera_vector)
    # up_vector = rot_matrix.dot(init_up_vector)
    # view_matrix = p.computeViewMatrix(com_p, com_p + 0.1 * camera_vector, up_vector)
    # p.getCameraImage(128, 128, view_matrix, projection_matrix,renderer=p.ER_BULLET_HARDWARE_OPENGL)