from scipy.spatial.transform import Rotation

rotation = Rotation.from_euler('z', 180, degrees=True)
quaternion = rotation.as_quat()