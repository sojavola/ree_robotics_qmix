#!/usr/bin/env python3
"""
Mineral Heatmap Publisher
=========================
Publie des heatmaps detaillees pour chaque type de mineral REE.
Inclut des visualisations avancees: contours, gradients, et annotations.
"""

import rclpy
from rclpy.node import Node
import numpy as np
from scipy import ndimage

from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration


class MineralHeatmapPublisher(Node):
    """Publie des heatmaps detaillees des mineraux"""

    MINERAL_INFO = {
        0: {'name': 'REE_Oxides', 'color': (1.0, 0.0, 0.0), 'symbol': 'Ox'},
        1: {'name': 'REE_Silicates', 'color': (0.0, 1.0, 0.0), 'symbol': 'Si'},
        2: {'name': 'REE_Phosphates', 'color': (0.0, 0.0, 1.0), 'symbol': 'Ph'},
        3: {'name': 'REE_Carbonates', 'color': (1.0, 1.0, 0.0), 'symbol': 'Ca'},
    }

    def __init__(self):
        super().__init__('mineral_heatmap_publisher')

        # Parametres
        self.declare_parameter('map_width', 100)
        self.declare_parameter('map_height', 100)
        self.declare_parameter('cell_size', 1.0)
        self.declare_parameter('heatmap_resolution', 2)  # Sous-echantillonnage
        self.declare_parameter('contour_levels', 5)
        self.declare_parameter('show_peak_labels', True)

        self.map_width = self.get_parameter('map_width').value
        self.map_height = self.get_parameter('map_height').value
        self.cell_size = self.get_parameter('cell_size').value
        self.resolution = self.get_parameter('heatmap_resolution').value
        self.contour_levels = self.get_parameter('contour_levels').value
        self.show_peaks = self.get_parameter('show_peak_labels').value

        # Etat
        self.mineral_map = None

        # Subscriber
        self.mineral_sub = self.create_subscription(
            Float32MultiArray, '/mineral_map',
            self.mineral_callback, 10)

        # Publishers
        self.heatmap_pubs = {}
        for idx, info in self.MINERAL_INFO.items():
            self.heatmap_pubs[idx] = self.create_publisher(
                MarkerArray, f'/viz/heatmap/{info["name"].lower()}', 10)

        self.combined_pub = self.create_publisher(
            MarkerArray, '/viz/heatmap/combined', 10)

        self.contours_pub = self.create_publisher(
            MarkerArray, '/viz/mineral_contours', 10)

        self.peaks_pub = self.create_publisher(
            MarkerArray, '/viz/mineral_peaks', 10)

        self.gradient_pub = self.create_publisher(
            MarkerArray, '/viz/mineral_gradient', 10)

        # Timer
        self.create_timer(0.5, self.publish_heatmaps)

        self.get_logger().info('Mineral Heatmap Publisher started')

    def mineral_callback(self, msg: Float32MultiArray):
        """Callback pour la carte des mineraux"""
        try:
            data = np.array(msg.data)
            self.mineral_map = data.reshape((self.map_height, self.map_width, 4))
        except Exception as e:
            self.get_logger().error(f'Error parsing mineral map: {e}')

    def publish_heatmaps(self):
        """Publie toutes les heatmaps"""
        if self.mineral_map is None:
            return

        stamp = self.get_clock().now().to_msg()

        # Heatmaps individuelles
        for mineral_idx in range(4):
            self.publish_single_heatmap(mineral_idx, stamp)

        # Heatmap combinee
        self.publish_combined_heatmap(stamp)

        # Contours
        self.publish_contours(stamp)

        # Pics de concentration
        if self.show_peaks:
            self.publish_peaks(stamp)

        # Gradients
        self.publish_gradients(stamp)

    def publish_single_heatmap(self, mineral_idx: int, stamp):
        """Publie la heatmap d'un type de mineral"""
        markers = MarkerArray()
        info = self.MINERAL_INFO[mineral_idx]
        mineral_layer = self.mineral_map[:, :, mineral_idx]

        marker_id = 0
        step = self.resolution

        for y in range(0, self.map_height, step):
            for x in range(0, self.map_width, step):
                # Moyenne de la region
                y_end = min(y + step, self.map_height)
                x_end = min(x + step, self.map_width)
                concentration = float(np.mean(mineral_layer[y:y_end, x:x_end]))

                if concentration > 0.05:
                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.header.stamp = stamp
                    marker.ns = f'heatmap_{info["name"]}'
                    marker.id = marker_id
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD

                    marker.pose.position.x = (x + step/2) * self.cell_size
                    marker.pose.position.y = (y + step/2) * self.cell_size
                    marker.pose.position.z = concentration * 1.5
                    marker.pose.orientation.w = 1.0

                    marker.scale.x = self.cell_size * step * 0.95
                    marker.scale.y = self.cell_size * step * 0.95
                    marker.scale.z = max(0.1, concentration * 3.0)

                    r, g, b = info['color']
                    marker.color.r = r
                    marker.color.g = g
                    marker.color.b = b
                    marker.color.a = 0.3 + concentration * 0.6

                    marker.lifetime = Duration(sec=1, nanosec=0)
                    markers.markers.append(marker)
                    marker_id += 1

        self.heatmap_pubs[mineral_idx].publish(markers)

    def publish_combined_heatmap(self, stamp):
        """Publie une heatmap combinee avec tous les mineraux"""
        markers = MarkerArray()
        marker_id = 0
        step = self.resolution * 2  # Plus grande resolution pour la combinee

        for y in range(0, self.map_height, step):
            for x in range(0, self.map_width, step):
                y_end = min(y + step, self.map_height)
                x_end = min(x + step, self.map_width)

                concentrations = np.mean(self.mineral_map[y:y_end, x:x_end, :], axis=(0, 1))
                total_conc = float(np.sum(concentrations))

                if total_conc > 0.1:
                    dominant_idx = np.argmax(concentrations)
                    info = self.MINERAL_INFO[dominant_idx]

                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.header.stamp = stamp
                    marker.ns = 'combined_heatmap'
                    marker.id = marker_id
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD

                    marker.pose.position.x = (x + step/2) * self.cell_size
                    marker.pose.position.y = (y + step/2) * self.cell_size
                    marker.pose.position.z = total_conc * 0.5
                    marker.pose.orientation.w = 1.0

                    marker.scale.x = self.cell_size * step * 0.9
                    marker.scale.y = self.cell_size * step * 0.9
                    marker.scale.z = max(0.2, total_conc * 2.0)

                    r, g, b = info['color']
                    marker.color.r = r
                    marker.color.g = g
                    marker.color.b = b
                    marker.color.a = 0.4 + min(0.5, total_conc * 0.3)

                    marker.lifetime = Duration(sec=1, nanosec=0)
                    markers.markers.append(marker)
                    marker_id += 1

        self.combined_pub.publish(markers)

    def publish_contours(self, stamp):
        """Publie les contours de concentration"""
        markers = MarkerArray()
        marker_id = 0

        for mineral_idx in range(4):
            info = self.MINERAL_INFO[mineral_idx]
            mineral_layer = self.mineral_map[:, :, mineral_idx]

            # Niveaux de contour
            levels = np.linspace(0.2, 0.9, self.contour_levels)

            for level in levels:
                # Creer un masque pour ce niveau
                mask = mineral_layer >= level

                # Trouver les contours (simplification)
                edges = ndimage.binary_erosion(mask) ^ mask

                edge_points = np.argwhere(edges)

                if len(edge_points) < 3:
                    continue

                # Creer une ligne de contour
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.header.stamp = stamp
                marker.ns = f'contour_{info["name"]}_{level:.1f}'
                marker.id = marker_id
                marker.type = Marker.LINE_LIST
                marker.action = Marker.ADD

                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.15

                r, g, b = info['color']
                marker.color.r = r
                marker.color.g = g
                marker.color.b = b
                marker.color.a = 0.5 + level * 0.4

                # Connecter les points voisins
                for i, (y, x) in enumerate(edge_points[:-1]):
                    next_y, next_x = edge_points[i + 1]
                    dist = np.sqrt((next_y - y)**2 + (next_x - x)**2)

                    if dist < 3:  # Seulement connecter les points proches
                        p1 = Point(
                            x=float(x * self.cell_size),
                            y=float(y * self.cell_size),
                            z=float(level * 2.0)
                        )
                        p2 = Point(
                            x=float(next_x * self.cell_size),
                            y=float(next_y * self.cell_size),
                            z=float(level * 2.0)
                        )
                        marker.points.extend([p1, p2])

                if marker.points:
                    marker.lifetime = Duration(sec=1, nanosec=0)
                    markers.markers.append(marker)
                    marker_id += 1

        self.contours_pub.publish(markers)

    def publish_peaks(self, stamp):
        """Publie les pics de concentration avec labels"""
        markers = MarkerArray()
        marker_id = 0

        for mineral_idx in range(4):
            info = self.MINERAL_INFO[mineral_idx]
            mineral_layer = self.mineral_map[:, :, mineral_idx]

            # Trouver les maxima locaux
            max_filter = ndimage.maximum_filter(mineral_layer, size=10)
            peaks = (mineral_layer == max_filter) & (mineral_layer > 0.5)

            peak_coords = np.argwhere(peaks)

            for y, x in peak_coords[:10]:  # Limiter a 10 pics par type
                concentration = float(mineral_layer[y, x])

                # Sphere au pic
                sphere = Marker()
                sphere.header.frame_id = 'map'
                sphere.header.stamp = stamp
                sphere.ns = f'peak_{info["name"]}'
                sphere.id = marker_id
                sphere.type = Marker.SPHERE
                sphere.action = Marker.ADD

                sphere.pose.position.x = float(int(x) * self.cell_size)
                sphere.pose.position.y = float(int(y) * self.cell_size)
                sphere.pose.position.z = concentration * 3.0
                sphere.pose.orientation.w = 1.0

                size = 1.0 + concentration * 2.0
                sphere.scale.x = size
                sphere.scale.y = size
                sphere.scale.z = size

                r, g, b = info['color']
                sphere.color.r = r
                sphere.color.g = g
                sphere.color.b = b
                sphere.color.a = 0.8

                sphere.lifetime = Duration(sec=1, nanosec=0)
                markers.markers.append(sphere)
                marker_id += 1

                # Label avec concentration
                label = Marker()
                label.header.frame_id = 'map'
                label.header.stamp = stamp
                label.ns = f'peak_label_{info["name"]}'
                label.id = marker_id
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD

                label.pose.position.x = float(int(x) * self.cell_size)
                label.pose.position.y = float(int(y) * self.cell_size)
                label.pose.position.z = concentration * 3.0 + 2.0

                label.scale.z = 1.2

                label.color.r = 1.0
                label.color.g = 1.0
                label.color.b = 1.0
                label.color.a = 1.0

                label.text = f'{info["symbol"]}: {concentration:.0%}'
                label.lifetime = Duration(sec=1, nanosec=0)
                markers.markers.append(label)
                marker_id += 1

        self.peaks_pub.publish(markers)

    def publish_gradients(self, stamp):
        """Publie les vecteurs de gradient (direction vers haute concentration)"""
        markers = MarkerArray()
        marker_id = 0

        # Calculer le gradient total
        total_concentration = np.sum(self.mineral_map, axis=2)
        grad_y, grad_x = np.gradient(total_concentration)

        step = 8  # Un vecteur tous les 8 pixels

        for y in range(step, self.map_height - step, step):
            for x in range(step, self.map_width - step, step):
                gx = grad_x[y, x]
                gy = grad_y[y, x]
                magnitude = np.sqrt(gx**2 + gy**2)

                if magnitude > 0.05:
                    arrow = Marker()
                    arrow.header.frame_id = 'map'
                    arrow.header.stamp = stamp
                    arrow.ns = 'gradient'
                    arrow.id = marker_id
                    arrow.type = Marker.ARROW
                    arrow.action = Marker.ADD

                    start = Point(
                        x=float(x * self.cell_size),
                        y=float(y * self.cell_size),
                        z=0.2
                    )

                    # Normaliser et multiplier par la magnitude
                    scale = min(5.0, magnitude * 10.0)
                    end = Point(
                        x=float(x * self.cell_size + (gx / magnitude) * scale),
                        y=float(y * self.cell_size + (gy / magnitude) * scale),
                        z=0.2
                    )

                    arrow.points = [start, end]

                    arrow.scale.x = 0.3
                    arrow.scale.y = 0.5
                    arrow.scale.z = 0.5

                    # Couleur basee sur la magnitude
                    intensity = float(min(1.0, magnitude * 2.0))
                    arrow.color.r = intensity
                    arrow.color.g = 1.0 - intensity * 0.5
                    arrow.color.b = 0.0
                    arrow.color.a = 0.6

                    arrow.lifetime = Duration(sec=1, nanosec=0)
                    markers.markers.append(arrow)
                    marker_id += 1

        self.gradient_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)

    try:
        node = MineralHeatmapPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
