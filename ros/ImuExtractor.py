import roslib
import rosbag
import rospy
from sensor_msgs.msg import Imu
import numpy as np
import pandas as pd
import sys

class ImuExtractor(object):
  def __init__(self, rosbag_path, topic_name, acc_unit="m/s^2", gyr_unit="rad/s"):
    self.rosbag_path = rosbag_path
    self.topic_name = topic_name
    self.acc_unit = acc_unit
    self.gyr_unit = gyr_unit
    print("supported acc_unit: m/s^2, g; gyr_unit: rad/s, deg/s")

  def __call__(self):
    with rosbag.Bag(self.rosbag_path, 'r') as bag:
      for topic, msg, t in bag.read_messages():
        if topic == self.topic_name:
          ts = msg.header.stamp.to_sec()
          ax = msg.linear_acceleration.x
          ay = msg.linear_acceleration.y
          az = msg.linear_acceleration.z
          if self.acc_unit == "g":
            ax, ay, az = ax / 9.8, ay / 9.8, az / 9.8
          elif self.acc_unit != "m/s^2":
            print("unsupported acc unit: %s" % self.acc_unit)
          gx = msg.angular_velocity.x
          gy = msg.angular_velocity.y
          gz = msg.angular_velocity.z
          if self.gyr_unit == "deg/s":
            gyr_scale = 1.0 / (2 * 3.1415926) * 360
            gx, gy, gz = gx * gyr_scale, gy * gyr_scale, gz * gyr_scale 
          elif self.acc_unit != "rad/s":
            print("unsupported gyr unit: %s" % self.gyr_unit)
          #yield ts, ax, ay, az, gx, gy, gz
          yield ts, gx, gy, gz, ax, ay, az

def test(input_bag, output_csv, imu_topic):
  imu_extractor = ImuExtractor(input_bag, imu_topic, "g", "deg/s")
  columns = ["timestamp","Gyroscope_X_deg_s","Gyroscope_Y_deg_s","Gyroscope_Z_deg_s",
                         "Accelerometer_X_g","Accelerometer_Y_g","Accelerometer_Z_g",
                         "Magnetometer_X_G","Magnetometer_Y_G","Magnetometer_Z_G"] 
  data = list(imu_extractor())
  data = np.array(data, dtype=np.float64)
  mag_data = np.zeros([len(data), 3], dtype=np.float64)
  data = np.hstack([data, mag_data])
  df = pd.DataFrame(data, columns=columns)
  df.to_csv(output_csv, index=False)

if __name__ == "__main__":
  if len(sys.argv) == 1:
    input_bag = "imu_zupt.bag"
    output_csv = "imu_zupt.csv"
    imu_topic = "/camera/imu"
  elif len(sys.argv) == 4:
    input_bag = sys.argv[1]
    output_csv = sys.argv[2]
    imu_topic = sys.argv[3]
  else:
    print("python ImuExtractor.py input_bag output_csv")
    exit(-1)

  test(input_bag, output_csv, imu_topic)
