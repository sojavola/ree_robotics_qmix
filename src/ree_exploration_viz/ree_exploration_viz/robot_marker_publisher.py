#!/usr/bin/env python3
"""
Robot Marker Publisher
======================
Publie les markers des robots avec trajectoires et zones de detection.
"""

import rclpy
from rclpy.node import Node
import numpy as np
from collections import deque

from geometry_msgs.msg import Pose2D, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration


class RobotMarkerPublisher(Node):
    """Publie les markers des robots pour RViz"""

    ROBOT_COLORS = [
        (0.0, 0.5, 1.0),   # Robot 0: Bleu clair
        (0.0, 1.0, 0.5),   # Robot 1: Vert clair
        (1.0, 0.5, 0.0),   # Robot 2: Orange
        (1.0, 0.0, 0.5),   # Robot 3: Magenta
    ]

    def __init__(self):
        super().__init__('robot_marker_publisher')

        # Parametres
        self.declare_parameter('num_robots', 4)
        self.declare_parameter('cell_size', 1.0)
        self.declare_parameter('trail_length', 50)
        self.declare_parameter('detection_radius', 5.0)

        self.num_robots = self.get_parameter('num_robots').value
        self.cell_size = self.get_parameter('cell_size').value
        self.trail_length = self.get_parameter('trail_length').value
        self.detection_radius = self.get_parameter('detection_radius').value

        # Stockage des positions et trajectoires
        self.robot_positions = {}
        self.robot_trails = {i: deque(maxlen=self.trail_length) for i in range(self.num_robots)}

        # Subscribers
        for i in range(self.num_robots):
            self.create_subscription(
                Pose2D, f'/robot_{i}/position',
                self.create_position_callback(i), 10)

        # Publishers
        self.markers_pub = self.create_publisher(MarkerArray, '/viz/robots_detailed', 10)
        self.trails_pub = self.create_publisher(MarkerArray, '/viz/robot_trails', 10)
        self.detection_pub = self.create_publisher(MarkerArray, '/viz/detection_zones', 10)

        # Timer
        self.create_timer(0.1, self.publish_all)

        self.get_logger().info('Robot Marker Publisher started')

    def create_position_callback(self, robot_id: int):
        """Cree un callback pour la position d'un robot"""
        def callback(msg: Pose2D):
            pos = (msg.x * self.cell_size, msg.y * self.cell_size, msg.theta)
            self.robot_positions[robot_id] = pos
            self.robot_trails[robot_id].append((pos[0], pos[1]))
        return callback

    def publish_all(self):
        """Publie tous les markers"""
        stamp = self.get_clock().now().to_msg()
        self.publish_robot_markers(stamp)
        self.publish_trails(stamp)
        self.publish_detection_zones(stamp)

    def publish_robot_markers(self, stamp):
        """Publie les robots avec corps, direction et label"""
        markers = MarkerArray()

        for robot_id, (x, y, theta) in self.robot_positions.items():
            color = self.ROBOT_COLORS[robot_id % len(self.ROBOT_COLORS)]

            # Corps du robot (mesh ou cylindre)
            body = Marker()
            body.header.frame_id = 'map'
            body.header.stamp = stamp
            body.ns = 'robot_body'
            body.id = robot_id
            body.type = Marker.MESH_RESOURCE
            body.mesh_resource = "package://ree_exploration_viz/meshes/rover.dae"
            body.mesh_use_embedded_materials = False
            body.action = Marker.ADD

            # Fallback si mesh non disponible
            body.type = Marker.CYLINDER

            body.pose.position.x = x
            body.pose.position.y = y
            body.pose.position.z = 0.5

            # Orientation basee sur theta
            body.pose.orientation.z = np.sin(theta / 2)
            body.pose.orientation.w = np.cos(theta / 2)

            body.scale.x = 3.0
            body.scale.y = 3.0
            body.scale.z = 1.5

            body.color.r = color[0]
            body.color.g = color[1]
            body.color.b = color[2]
            body.color.a = 0.9

            body.lifetime = Duration(sec=0, nanosec=200000000)
            markers.markers.append(body)

            # Fleche de direction
            arrow = Marker()
            arrow.header.frame_id = 'map'
            arrow.header.stamp = stamp
            arrow.ns = 'robot_arrow'
            arrow.id = robot_id
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD

            start = Point(x=x, y=y, z=1.5)
            end = Point(
                x=x + np.cos(theta) * 4.0,
                y=y + np.sin(theta) * 4.0,
                z=1.5
            )
            arrow.points = [start, end]

            arrow.scale.x = 0.5
            arrow.scale.y = 0.8
            arrow.scale.z = 1.0

            arrow.color.r = 1.0
            arrow.color.g = 1.0
            arrow.color.b = 1.0
            arrow.color.a = 1.0

            arrow.lifetime = Duration(sec=0, nanosec=200000000)
            markers.markers.append(arrow)

            # Label avec ID et coordonnees
            label = Marker()
            label.header.frame_id = 'map'
            label.header.stamp = stamp
            label.ns = 'robot_label'
            label.id = robot_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD

            label.pose.position.x = x
            label.pose.position.y = y
            label.pose.position.z = 3.5

            label.scale.z = 1.8

            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.color.a = 1.0

            label.text = f'R{robot_id}\n({x/self.cell_size:.0f},{y/self.cell_size:.0f})'
            label.lifetime = Duration(sec=0, nanosec=200000000)
            markers.markers.append(label)

        self.markers_pub.publish(markers)

    def publish_trails(self, stamp):
        """Publie les trajectoires des robots"""
        markers = MarkerArray()

        for robot_id, trail in self.robot_trails.items():
            if len(trail) < 2:
                continue

            color = self.ROBOT_COLORS[robot_id % len(self.ROBOT_COLORS)]

            line = Marker()
            line.header.frame_id = 'map'
            line.header.stamp = stamp
            line.ns = 'robot_trail'
            line.id = robot_id
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD

            line.pose.orientation.w = 1.0
            line.scale.x = 0.3

            # Points de la trajectoire avec gradient de couleur
            for i, (x, y) in enumerate(trail):
                point = Point(x=x, y=y, z=0.1)
                line.points.append(point)

                # Couleur avec fade
                alpha = 0.2 + 0.8 * (i / len(trail))
                c = ColorRGBA(r=color[0], g=color[1], b=color[2], a=alpha)
                line.colors.append(c)

            line.lifetime = Duration(sec=1, nanosec=0)
            markers.markers.append(line)

        self.trails_pub.publish(markers)

    def publish_detection_zones(self, stamp):
        """Publie les zones de detection des robots"""
        markers = MarkerArray()

        for robot_id, (x, y, _) in self.robot_positions.items():
            color = self.ROBOT_COLORS[robot_id % len(self.ROBOT_COLORS)]

            # Cercle de detection
            circle = Marker()
            circle.header.frame_id = 'map'
            circle.header.stamp = stamp
            circle.ns = 'detection_zone'
            circle.id = robot_id
            circle.type = Marker.CYLINDER
            circle.action = Marker.ADD

            circle.pose.position.x = x
            circle.pose.position.y = y
            circle.pose.position.z = 0.05
            circle.pose.orientation.w = 1.0

            circle.scale.x = self.detection_radius * 2
            circle.scale.y = self.detection_radius * 2
            circle.scale.z = 0.1

            circle.color.r = color[0]
            circle.color.g = color[1]
            circle.color.b = color[2]
            circle.color.a = 0.15

            circle.lifetime = Duration(sec=0, nanosec=200000000)
            markers.markers.append(circle)

        self.detection_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)

    try:
        node = RobotMarkerPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
