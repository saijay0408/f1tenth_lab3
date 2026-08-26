#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollow(Node):
    def __init__(self):
        super().__init__('wall_follow_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        self.subscription = self.create_subscription(
            LaserScan, lidarscan_topic, self.scan_callback, 10)
        self.publisher_ = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10)

        self.kp = 1.1
        self.kd = 0.6
        self.ki = 0.0

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

        self.desired_distance = 0.9
        self.lookahead = 0.6

        self.theta = np.radians(50.0)
        self.angle_b = np.radians(90.0)
        self.angle_a = self.angle_b - self.theta

        self.angle_min = None
        self.angle_increment = None
        self.range_max = 10.0

        self.max_steer = np.radians(24.0)

    def get_range(self, range_data, angle):
        index = int(round((angle - self.angle_min) / self.angle_increment))
        index = int(np.clip(index, 0, len(range_data) - 1))
        r = range_data[index]

        if np.isnan(r):
            return 0.0
        if np.isinf(r):
            return self.range_max
        return float(r)

    def get_error(self, range_data, dist):
        a = self.get_range(range_data, self.angle_a)
        b = self.get_range(range_data, self.angle_b)

        alpha = np.arctan2(a * np.cos(self.theta) - b, a * np.sin(self.theta))
        current_dist = b * np.cos(alpha)
        projected_dist = current_dist + self.lookahead * np.sin(alpha)

        return projected_dist - dist

    def get_velocity(self, angle_estimate):
        abs_angle_deg = abs(np.degrees(angle_estimate))
        if abs_angle_deg <= 10.0:
            return 1.5
        elif abs_angle_deg <= 20.0:
            return 1.0
        else:
            return 0.5

    def pid_control(self, error, velocity):
        now = self.get_clock().now().nanoseconds / 1e9
        dt = 0.0 if self.prev_time is None else max(now - self.prev_time, 1e-4)
        self.prev_time = now

        self.integral += error * dt
        derivative = 0.0 if dt == 0.0 else (error - self.prev_error) / dt
        self.prev_error = error

        angle = self.kp * error + self.ki * self.integral + self.kd * derivative
        angle = float(np.clip(angle, -self.max_steer, self.max_steer))

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = angle
        drive_msg.drive.speed = velocity
        self.publisher_.publish(drive_msg)

    def scan_callback(self, msg):
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment
        self.range_max = msg.range_max
        range_data = msg.ranges

        error = self.get_error(range_data, self.desired_distance)
        angle_estimate = self.kp * error
        velocity = self.get_velocity(angle_estimate)
        self.pid_control(error, velocity)


def main(args=None):
    rclpy.init(args=args)
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)

    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
