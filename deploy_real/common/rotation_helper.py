import numpy as np
from scipy.spatial.transform import Rotation as R


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
    R_pelvis = np.dot(R_torso, RzWaist.T)
    w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
    return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w


def transform_pelvis_to_torso_complete(waist_yaw, waist_roll, waist_pitch, pelvis_quat):
    """
    完整的pelvis到torso变换，包含所有三个waist关节
    """
    from scipy.spatial.transform import Rotation as R
    
    # 按照机器人运动链的顺序: pelvis -> waist_yaw -> waist_roll -> waist_pitch -> torso
    R_waist_yaw = R.from_euler("z", waist_yaw)     # 绕Z轴
    R_waist_roll = R.from_euler("x", waist_roll)   # 绕X轴  
    R_waist_pitch = R.from_euler("y", waist_pitch) # 绕Y轴
    
    # pelvis的旋转
    R_pelvis = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]])
    
    # 组合旋转: torso = pelvis * waist_yaw * waist_roll * waist_pitch
    R_torso = R_pelvis * R_waist_yaw * R_waist_roll * R_waist_pitch
    
    # 转换为wxyz格式
    torso_quat_scipy = R_torso.as_quat()  # [x,y,z,w]格式
    torso_quat = np.array([torso_quat_scipy[3], torso_quat_scipy[0], torso_quat_scipy[1], torso_quat_scipy[2]])
    
    return torso_quat
