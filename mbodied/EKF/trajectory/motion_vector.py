import numpy as np
from mbodied.types.motion.control import (
    LocationAngle,
    Pose,
    JointControl,
    FullJointControl,
    HandControl,
    HeadControl,
    MobileSingleHandControl,
    MobileSingleArmControl,
    MobileBimanualArmControl,
    HumanoidControl
)


class LocationAngleVector:
    def __init__(self, base: LocationAngle):
        self.base = base
    
    def to_vector(self):
        return np.array([
            self.base.x,
            self.base.y,
            self.base.theta
        ])


class PoseVector:
    def __init__(self, pose: Pose):
        self.pose = pose

    def to_vector(self):
        return np.array([
            self.pose.x,  
            self.pose.y,
            self.pose.z,  
            self.pose.roll, 
            self.pose.pitch,
            self.pose.yaw
        ])


class JointControlVector:
    def __init__(self, joint_control: JointControl):
        self.joint_control = joint_control

    def to_vector(self):
        return np.array([self.joint_control.value])


class FullJointControlVector:
    def __init__(self, full_joint_control: FullJointControl):
        self.full_joint_control = full_joint_control

    def to_vector(self):
        joint_values = [joint.value for joint in self.full_joint_control.joints]
        return np.array(joint_values)


class HandControlVector:
    def __init__(self, hand_control: HandControl):
        self.hand_control = hand_control

    def to_vector(self):
        pose_vector = PoseVector(self.hand_control.pose).to_vector()
        grasp_value = np.array([self.hand_control.grasp.value])
        return np.concatenate([pose_vector, grasp_value])


class HeadControlVector:
    def __init__(self, head_control: HeadControl):
        self.head_control = head_control

    def to_vector(self):
        return np.array([
            self.head_control.tilt.value,
            self.head_control.pan.value
        ])


class MobileSingleHandControlVector:
    def __init__(self, mobile_control: MobileSingleHandControl):
        self.mobile_control = mobile_control

    def to_vector(self):
        base_vector = LocationAngleVector(self.mobile_control.base).to_vector() if self.mobile_control.base else np.array([])
        hand_vector = HandControlVector(self.mobile_control.hand).to_vector() if self.mobile_control.hand else np.array([])
        head_vector = HeadControlVector(self.mobile_control.head).to_vector() if self.mobile_control.head else np.array([])

        return np.concatenate([base_vector, hand_vector, head_vector])


class MobileSingleArmControlVector:
    def __init__(self, mobile_control: MobileSingleArmControl):
        self.mobile_control = mobile_control

    def to_vector(self):
        base_vector = LocationAngleVector(self.mobile_control.base).to_vector() if self.mobile_control.base else np.array([])
        arm_vector = FullJointControlVector(self.mobile_control.arm).to_vector() if self.mobile_control.arm else np.array([])
        head_vector = HeadControlVector(self.mobile_control.head).to_vector() if self.mobile_control.head else np.array([])

        return np.concatenate([base_vector, arm_vector, head_vector])


class MobileBimanualArmControlVector:
    def __init__(self, mobile_bimanual_control: MobileBimanualArmControl):
        self.mobile_bimanual_control = mobile_bimanual_control

    def to_vector(self):
        base_vector = LocationAngleVector(self.mobile_bimanual_control.base).to_vector() if self.mobile_bimanual_control.base else np.array([])
        left_arm_vector = FullJointControlVector(self.mobile_bimanual_control.left_arm).to_vector() if self.mobile_bimanual_control.left_arm else np.array([])
        right_arm_vector = FullJointControlVector(self.mobile_bimanual_control.right_arm).to_vector() if self.mobile_bimanual_control.right_arm else np.array([])
        head_vector = HeadControlVector(self.mobile_bimanual_control.head).to_vector() if self.mobile_bimanual_control.head else np.array([])

        return np.concatenate([base_vector, left_arm_vector, right_arm_vector, head_vector])


class HumanoidControlVector:
    def __init__(self, humanoid_control: HumanoidControl):
        self.humanoid_control = humanoid_control

    def to_vector(self):
        left_arm_vector = FullJointControlVector(self.humanoid_control.left_arm).to_vector() if self.humanoid_control.left_arm else np.array([])
        right_arm_vector = FullJointControlVector(self.humanoid_control.right_arm).to_vector() if self.humanoid_control.right_arm else np.array([])
        left_leg_vector = FullJointControlVector(self.humanoid_control.left_leg).to_vector() if self.humanoid_control.left_leg else np.array([])
        right_leg_vector = FullJointControlVector(self.humanoid_control.right_leg).to_vector() if self.humanoid_control.right_leg else np.array([])
        head_vector = HeadControlVector(self.humanoid_control.head).to_vector() if self.humanoid_control.head else np.array([])

        return np.concatenate([left_arm_vector, right_arm_vector, left_leg_vector, right_leg_vector, head_vector])
