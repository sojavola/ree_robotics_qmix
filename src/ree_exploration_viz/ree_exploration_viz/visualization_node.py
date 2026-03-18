#!/usr/bin/env python3
"""
REE Exploration Visualization Node
===================================
Node principal de visualisation RViz2 pour le systeme d'exploration REE.
Convertit les donnees du serveur en markers, heatmaps, et pointclouds pour RViz2.

Visualisations:
- Carte des mineraux (4 types REE) en PointCloud2 colore
- Carte des obstacles en GridCells
- Heatmap de concentration en MarkerArray (cubes)
- Veines minerales en LineStrip markers
- Zones explorees en overlay transparent
- Couches souterraines avec transparence variable
- Clusters detectes en spheres
"""

import rclpy
from rclpy.node import Node
import numpy as np
from typing import List, Tuple, Dict
import struct

# Messages ROS2
from std_msgs.msg import Float32MultiArray, String, Header, ColorRGBA
from geometry_msgs.msg import Pose2D, Point, Vector3, Quaternion, Pose
from nav_msgs.msg import OccupancyGrid, GridCells
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration


class REEVisualizationNode(Node):
    """Node de visualisation RViz2 pour l'exploration REE"""

    # Couleurs géologiques naturelles des REE (RGBA)
    MINERAL_COLORS = {
        0: {'name': 'REE_Oxides',     'color': (0.95, 0.35, 0.10, 0.85)},  # Orange-rouge (goethite)
        1: {'name': 'REE_Silicates',  'color': (0.20, 0.72, 0.30, 0.85)},  # Vert forêt (malachite)
        2: {'name': 'REE_Phosphates', 'color': (0.55, 0.20, 0.90, 0.85)},  # Violet (apatite)
        3: {'name': 'REE_Carbonates', 'color': (0.90, 0.72, 0.10, 0.85)},  # Ambre/or (calcite)
    }

    # Couleurs des robots (vives et distinctes)
    ROBOT_COLORS = [
        (0.0,  0.75, 1.0,  1.0),   # Robot 0: Cyan brillant
        (0.1,  0.90, 0.1,  1.0),   # Robot 1: Vert lime
        (1.0,  0.45, 0.0,  1.0),   # Robot 2: Orange vif
        (0.90, 0.0,  0.90, 1.0),   # Robot 3: Magenta vif
    ]

    def __init__(self):
        super().__init__('ree_visualization_node')

        # === PARAMETRES ===
        self.declare_parameter('map_width', 100)
        self.declare_parameter('map_height', 100)
        self.declare_parameter('num_robots', 4)
        self.declare_parameter('cell_size', 1.0)  # Taille d'une cellule en metres
        self.declare_parameter('visualization_rate', 5.0)  # Hz
        self.declare_parameter('mineral_threshold', 0.30)  # Seuil de visualisation
        self.declare_parameter('show_underground', True)
        self.declare_parameter('show_veins', True)
        self.declare_parameter('show_clusters', True)

        # Recuperer les parametres
        self.map_width = self.get_parameter('map_width').value
        self.map_height = self.get_parameter('map_height').value
        self.num_robots = self.get_parameter('num_robots').value
        self.cell_size = self.get_parameter('cell_size').value
        self.viz_rate = self.get_parameter('visualization_rate').value
        self.mineral_threshold = self.get_parameter('mineral_threshold').value
        self.show_underground = self.get_parameter('show_underground').value
        self.show_veins = self.get_parameter('show_veins').value
        self.show_clusters = self.get_parameter('show_clusters').value

        # === ETAT INTERNE ===
        self.mineral_map = None
        self.underground_layers = None
        self.obstacle_map = None
        self.robot_positions = {}
        self.exploration_map = None

        # === SUBSCRIBERS ===
        self.mineral_sub = self.create_subscription(
            Float32MultiArray, '/mineral_map', self.mineral_callback, 10)

        self.obstacle_sub = self.create_subscription(
            OccupancyGrid, '/obstacle_map', self.obstacle_callback, 10)

        self.underground_sub = self.create_subscription(
            Float32MultiArray, '/underground_layers', self.underground_callback, 10)

        self.science_sub = self.create_subscription(
            Float32MultiArray, '/science_targets', self.science_callback, 10)

        # Positions des robots
        for i in range(self.num_robots):
            self.create_subscription(
                Pose2D, f'/robot_{i}/position',
                self.create_robot_callback(i), 10)

        # === PUBLISHERS RVIZ ===
        # Mineraux en PointCloud2
        self.mineral_pc_pub = self.create_publisher(
            PointCloud2, '/viz/mineral_pointcloud', 10)

        # Mineraux par type (4 publishers)
        self.mineral_type_pubs = {}
        for idx, info in self.MINERAL_COLORS.items():
            self.mineral_type_pubs[idx] = self.create_publisher(
                PointCloud2, f'/viz/minerals/{info["name"].lower()}', 10)

        # Heatmap en MarkerArray
        self.heatmap_pub = self.create_publisher(
            MarkerArray, '/viz/mineral_heatmap', 10)

        # Obstacles en GridCells
        self.obstacle_viz_pub = self.create_publisher(
            GridCells, '/viz/obstacles', 10)

        # Obstacles en MarkerArray (cubes)
        self.obstacle_marker_pub = self.create_publisher(
            MarkerArray, '/viz/obstacle_markers', 10)

        # Veines minerales
        self.veins_pub = self.create_publisher(
            MarkerArray, '/viz/mineral_veins', 10)

        # Clusters detectes
        self.clusters_pub = self.create_publisher(
            MarkerArray, '/viz/mineral_clusters', 10)

        # Robots
        self.robot_markers_pub = self.create_publisher(
            MarkerArray, '/viz/robots', 10)

        # Zone exploree
        self.exploration_pub = self.create_publisher(
            MarkerArray, '/viz/exploration_zone', 10)

        # Couches souterraines
        self.underground_pub = self.create_publisher(
            MarkerArray, '/viz/underground', 10)

        # Cibles scientifiques
        self.science_markers_pub = self.create_publisher(
            MarkerArray, '/viz/science_targets', 10)

        # Grille de reference
        self.grid_pub = self.create_publisher(
            MarkerArray, '/viz/reference_grid', 10)

        # Legende
        self.legend_pub = self.create_publisher(
            MarkerArray, '/viz/legend', 10)

        # === TIMERS ===
        period = 1.0 / self.viz_rate
        self.viz_timer = self.create_timer(period, self.publish_visualizations)
        self.grid_timer = self.create_timer(5.0, self.publish_reference_grid)
        self.legend_timer = self.create_timer(10.0, self.publish_legend)

        # Publier la grille et la legende au demarrage
        self.create_timer(1.0, self.initial_publish, callback_group=None)

        self.get_logger().info('=' * 60)
        self.get_logger().info('REE Visualization Node initialized')
        self.get_logger().info(f'  Map: {self.map_width}x{self.map_height}')
        self.get_logger().info(f'  Cell size: {self.cell_size}m')
        self.get_logger().info(f'  Visualization rate: {self.viz_rate}Hz')
        self.get_logger().info(f'  Mineral threshold: {self.mineral_threshold}')
        self.get_logger().info('=' * 60)

    def initial_publish(self):
        """Publication initiale"""
        self.publish_reference_grid()
        self.publish_legend()
        self.destroy_timer(self._timers[-1])  # Supprimer ce timer one-shot

    # === CALLBACKS ===

    def mineral_callback(self, msg: Float32MultiArray):
        """Callback pour la carte des mineraux"""
        try:
            data = np.array(msg.data)
            self.mineral_map = data.reshape((self.map_height, self.map_width, 4))
        except Exception as e:
            self.get_logger().error(f'Error parsing mineral map: {e}')

    def obstacle_callback(self, msg: OccupancyGrid):
        """Callback pour la carte des obstacles"""
        try:
            data = np.array(msg.data)
            self.obstacle_map = data.reshape((msg.info.height, msg.info.width))
        except Exception as e:
            self.get_logger().error(f'Error parsing obstacle map: {e}')

    def underground_callback(self, msg: Float32MultiArray):
        """Callback pour les couches souterraines"""
        try:
            data = np.array(msg.data)
            self.underground_layers = data.reshape((self.map_height, self.map_width, 4))
        except Exception as e:
            self.get_logger().error(f'Error parsing underground layers: {e}')

    def science_callback(self, msg: Float32MultiArray):
        """Callback pour les cibles scientifiques"""
        try:
            data = np.array(msg.data)
            self.science_targets = data.reshape((self.map_height, self.map_width))
        except Exception as e:
            self.get_logger().error(f'Error parsing science targets: {e}')

    def create_robot_callback(self, robot_id: int):
        """Cree un callback pour la position d'un robot"""
        def callback(msg: Pose2D):
            self.robot_positions[robot_id] = (msg.x, msg.y, msg.theta)
        return callback

    # === PUBLICATION DES VISUALISATIONS ===

    def publish_visualizations(self):
        """Publie toutes les visualisations"""
        stamp = self.get_clock().now().to_msg()

        # Mineraux — heatmap uniquement (pointcloud desactive pour eviter double rendu)
        if self.mineral_map is not None:
            self.publish_mineral_heatmap(stamp)

            if self.show_veins:
                self.publish_mineral_veins(stamp)

            if self.show_clusters:
                self.publish_mineral_clusters(stamp)

        # Obstacles
        if self.obstacle_map is not None:
            self.publish_obstacle_markers(stamp)

        # Robots
        if self.robot_positions:
            self.publish_robot_markers(stamp)

        # Couches souterraines
        if self.show_underground and self.underground_layers is not None:
            self.publish_underground_markers(stamp)

        # Cibles scientifiques
        if hasattr(self, 'science_targets') and self.science_targets is not None:
            self.publish_science_markers(stamp)

    def publish_mineral_pointcloud(self, stamp):
        """Publie les mineraux en PointCloud2 colore"""
        if self.mineral_map is None:
            return

        # Creer un PointCloud2 combine
        points = []
        colors = []

        for y in range(self.map_height):
            for x in range(self.map_width):
                for mineral_idx in range(4):
                    concentration = self.mineral_map[y, x, mineral_idx]
                    if concentration > self.mineral_threshold:
                        # Position
                        px = x * self.cell_size
                        py = y * self.cell_size
                        pz = concentration * 2.0  # Hauteur proportionnelle

                        points.append((px, py, pz))

                        # Couleur
                        color = self.MINERAL_COLORS[mineral_idx]['color']
                        r, g, b, a = color
                        a = min(1.0, concentration + 0.3)  # Alpha variable
                        colors.append((r, g, b, a))

        if points:
            pc_msg = self.create_pointcloud2(points, colors, stamp)
            self.mineral_pc_pub.publish(pc_msg)

        # Publier aussi par type de mineral
        for mineral_idx in range(4):
            type_points = []
            type_colors = []

            for y in range(self.map_height):
                for x in range(self.map_width):
                    concentration = self.mineral_map[y, x, mineral_idx]
                    if concentration > self.mineral_threshold:
                        px = x * self.cell_size
                        py = y * self.cell_size
                        pz = concentration * 2.0

                        type_points.append((px, py, pz))

                        color = self.MINERAL_COLORS[mineral_idx]['color']
                        r, g, b, _ = color
                        a = min(1.0, concentration + 0.3)
                        type_colors.append((r, g, b, a))

            if type_points:
                pc_msg = self.create_pointcloud2(type_points, type_colors, stamp)
                self.mineral_type_pubs[mineral_idx].publish(pc_msg)

    def create_pointcloud2(self, points: List[Tuple], colors: List[Tuple], stamp) -> PointCloud2:
        """Cree un message PointCloud2 avec couleurs RGBA"""
        header = Header()
        header.stamp = stamp
        header.frame_id = 'map'

        # Champs du PointCloud
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgba', offset=12, datatype=PointField.UINT32, count=1),
        ]

        # Creer les donnees binaires
        point_step = 16  # 4 floats * 4 bytes
        data = bytearray()

        for (px, py, pz), (r, g, b, a) in zip(points, colors):
            # Position
            data.extend(struct.pack('fff', px, py, pz))

            # Couleur RGBA (packed as uint32)
            rgba = (int(a * 255) << 24) | (int(b * 255) << 16) | (int(g * 255) << 8) | int(r * 255)
            data.extend(struct.pack('I', rgba))

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.row_step = point_step * len(points)
        msg.data = bytes(data)
        msg.is_dense = True

        return msg

    def publish_mineral_heatmap(self, stamp):
        """Publie les mineraux en petites spheres colorées (style point cloud)"""
        if self.mineral_map is None:
            return

        markers = MarkerArray()
        marker_id = 0

        # Afficher 1 cellule sur 2 (damier) pour réduire l'aspect grille
        step = 1
        for y in range(0, self.map_height, step):
            for x in range(0, self.map_width, step):
                concentrations = self.mineral_map[y, x, :]
                max_concentration = float(np.max(concentrations))

                if max_concentration > self.mineral_threshold:
                    dominant_idx = int(np.argmax(concentrations))

                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.header.stamp = stamp
                    marker.ns = 'mineral_heatmap'
                    marker.id = marker_id
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD

                    # Position — Z fixe (pas de dôme)
                    marker.pose.position.x = float(x * self.cell_size + self.cell_size / 2)
                    marker.pose.position.y = float(y * self.cell_size + self.cell_size / 2)
                    marker.pose.position.z = 0.12
                    marker.pose.orientation.w = 1.0

                    # Taille proportionnelle à la concentration
                    sphere_size = 0.45 + max_concentration * 0.30
                    marker.scale.x = sphere_size
                    marker.scale.y = sphere_size
                    marker.scale.z = sphere_size * 0.6

                    # Couleur géologique — alpha fort pour bien les voir
                    r, g, b, _ = self.MINERAL_COLORS[dominant_idx]['color']
                    marker.color.r = r
                    marker.color.g = g
                    marker.color.b = b
                    marker.color.a = min(1.0, 0.70 + max_concentration * 0.30)

                    marker.lifetime = Duration(sec=0, nanosec=0)
                    markers.markers.append(marker)
                    marker_id += 1

        self.heatmap_pub.publish(markers)

    def publish_mineral_veins(self, stamp):
        """Publie les veines minerales en LineStrip"""
        if self.mineral_map is None:
            return

        markers = MarkerArray()
        marker_id = 0

        for mineral_idx in range(4):
            mineral_layer = self.mineral_map[:, :, mineral_idx]

            # Detecter les contours des veines (zones > seuil)
            threshold = 0.4
            vein_mask = mineral_layer > threshold

            # Trouver les points de veine
            vein_points = np.argwhere(vein_mask)

            if len(vein_points) < 10:
                continue

            # Creer une ligne connectant les points de veine (simplifiee)
            # On prend un echantillon des points pour eviter trop de markers
            sample_size = min(100, len(vein_points))
            # Seed fixe par mineral_idx → mêmes points sélectionnés à chaque frame
            rng = np.random.default_rng(seed=mineral_idx)
            sampled_indices = rng.choice(len(vein_points), sample_size, replace=False)
            sampled_points = vein_points[sampled_indices]

            # Trier par position pour creer une ligne continue
            sorted_indices = np.lexsort((sampled_points[:, 1], sampled_points[:, 0]))
            sorted_points = sampled_points[sorted_indices]

            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = stamp
            marker.ns = f'vein_{self.MINERAL_COLORS[mineral_idx]["name"]}'
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.3  # Epaisseur de la ligne

            # Couleur de la veine
            r, g, b, _ = self.MINERAL_COLORS[mineral_idx]['color']
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 0.7

            # Ajouter les points
            for y, x in sorted_points:
                point = Point()
                point.x = float(x * self.cell_size)
                point.y = float(y * self.cell_size)
                point.z = float(mineral_layer[y, x] * 1.5)
                marker.points.append(point)

            marker.lifetime = Duration(sec=0, nanosec=0)
            markers.markers.append(marker)
            marker_id += 1

        self.veins_pub.publish(markers)

    def publish_mineral_clusters(self, stamp):
        """Publie les clusters mineraux detectes en spheres"""
        if self.mineral_map is None:
            return

        markers = MarkerArray()
        marker_id = 0

        for mineral_idx in range(4):
            mineral_layer = self.mineral_map[:, :, mineral_idx]

            # Detecter les clusters simples (zones de forte concentration)
            threshold = 0.5
            high_conc_mask = mineral_layer > threshold

            # Trouver les centres des clusters (zones connectees)
            from scipy import ndimage
            labeled, num_features = ndimage.label(high_conc_mask)

            for cluster_id in range(1, num_features + 1):
                cluster_mask = labeled == cluster_id
                cluster_points = np.argwhere(cluster_mask)

                if len(cluster_points) < 5:
                    continue

                # Centre du cluster
                center_y = float(np.mean(cluster_points[:, 0]))
                center_x = float(np.mean(cluster_points[:, 1]))

                # Concentration moyenne
                avg_concentration = float(np.mean(mineral_layer[cluster_mask]))

                # Taille du cluster
                cluster_size = float(np.sqrt(len(cluster_points)) * self.cell_size * 0.5)

                marker = Marker()
                marker.header.frame_id = 'map'
                marker.header.stamp = stamp
                marker.ns = f'cluster_{self.MINERAL_COLORS[mineral_idx]["name"]}'
                marker.id = marker_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD

                marker.pose.position.x = center_x * self.cell_size
                marker.pose.position.y = center_y * self.cell_size
                marker.pose.position.z = avg_concentration * 2.0
                marker.pose.orientation.w = 1.0

                marker.scale.x = cluster_size
                marker.scale.y = cluster_size
                marker.scale.z = cluster_size

                # Couleur avec transparence
                r, g, b, _ = self.MINERAL_COLORS[mineral_idx]['color']
                marker.color.r = r
                marker.color.g = g
                marker.color.b = b
                marker.color.a = 0.4

                marker.lifetime = Duration(sec=0, nanosec=0)
                markers.markers.append(marker)
                marker_id += 1

        self.clusters_pub.publish(markers)

    def publish_obstacle_markers(self, stamp):
        """Publie les obstacles en MarkerArray"""
        if self.obstacle_map is None:
            return

        markers = MarkerArray()
        marker_id = 0

        # Trouver les cellules obstacles
        obstacle_cells = np.argwhere(self.obstacle_map > 50)

        # Grouper les obstacles en blocs pour optimisation
        step = 2
        for y in range(0, self.map_height, step):
            for x in range(0, self.map_width, step):
                # Verifier si cette zone contient des obstacles
                y_end = min(y + step, self.map_height)
                x_end = min(x + step, self.map_width)
                region = self.obstacle_map[y:y_end, x:x_end]

                if np.any(region > 50):
                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.header.stamp = stamp
                    marker.ns = 'obstacles'
                    marker.id = marker_id
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD

                    marker.pose.position.x = (x + step / 2) * self.cell_size
                    marker.pose.position.y = (y + step / 2) * self.cell_size
                    marker.pose.position.z = 0.75
                    marker.pose.orientation.w = 1.0

                    marker.scale.x = self.cell_size * step * 0.95
                    marker.scale.y = self.cell_size * step * 0.95
                    marker.scale.z = 1.5

                    # Couleur roche naturelle (brun-gris) avec légère variation
                    variation = ((x * 7 + y * 13) % 10) / 100.0
                    marker.color.r = 0.38 + variation
                    marker.color.g = 0.30 + variation * 0.5
                    marker.color.b = 0.22 + variation * 0.3
                    marker.color.a = 0.95

                    marker.lifetime = Duration(sec=0, nanosec=0)
                    markers.markers.append(marker)
                    marker_id += 1

        self.obstacle_marker_pub.publish(markers)

    def publish_robot_markers(self, stamp):
        """Publie les positions des robots"""
        markers = MarkerArray()

        for robot_id, (x, y, theta) in self.robot_positions.items():
            # Corps du robot (cylindre)
            body = Marker()
            body.header.frame_id = 'map'
            body.header.stamp = stamp
            body.ns = 'robot_body'
            body.id = robot_id
            body.type = Marker.CYLINDER
            body.action = Marker.ADD

            body.pose.position.x = x * self.cell_size
            body.pose.position.y = y * self.cell_size
            body.pose.position.z = 0.5
            body.pose.orientation.w = 1.0

            body.scale.x = 1.8
            body.scale.y = 1.8
            body.scale.z = 0.8

            r, g, b, a = self.ROBOT_COLORS[robot_id % len(self.ROBOT_COLORS)]
            body.color.r = r
            body.color.g = g
            body.color.b = b
            body.color.a = a

            body.lifetime = Duration(sec=0, nanosec=500000000)
            markers.markers.append(body)

            # Fleche de direction
            arrow = Marker()
            arrow.header.frame_id = 'map'
            arrow.header.stamp = stamp
            arrow.ns = 'robot_direction'
            arrow.id = robot_id
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD

            # Point de depart
            start = Point()
            start.x = x * self.cell_size
            start.y = y * self.cell_size
            start.z = 1.2

            # Point d'arrivee (direction)
            end = Point()
            end.x = start.x + np.cos(theta) * 3.0
            end.y = start.y + np.sin(theta) * 3.0
            end.z = 1.2

            arrow.points = [start, end]
            arrow.scale.x = 0.4  # Diametre de la tige
            arrow.scale.y = 0.6  # Diametre de la tete
            arrow.scale.z = 0.8  # Longueur de la tete

            arrow.color.r = 1.0
            arrow.color.g = 1.0
            arrow.color.b = 1.0
            arrow.color.a = 1.0

            arrow.lifetime = Duration(sec=0, nanosec=500000000)
            markers.markers.append(arrow)

            # Label du robot
            label = Marker()
            label.header.frame_id = 'map'
            label.header.stamp = stamp
            label.ns = 'robot_label'
            label.id = robot_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD

            label.pose.position.x = x * self.cell_size
            label.pose.position.y = y * self.cell_size
            label.pose.position.z = 2.5

            label.scale.z = 1.5  # Taille du texte

            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.color.a = 1.0

            label.text = f'R{robot_id}'
            label.lifetime = Duration(sec=0, nanosec=500000000)
            markers.markers.append(label)

        self.robot_markers_pub.publish(markers)

    def publish_underground_markers(self, stamp):
        """Publie les couches souterraines"""
        if self.underground_layers is None:
            return

        markers = MarkerArray()
        marker_id = 0

        # Sous-echantillonner pour performance
        step = 4
        z_offset = -2.0  # Profondeur de la couche souterraine

        for y in range(0, self.map_height, step):
            for x in range(0, self.map_width, step):
                concentrations = self.underground_layers[y, x, :]
                max_conc = np.max(concentrations)

                if max_conc > self.mineral_threshold:
                    dominant_idx = np.argmax(concentrations)

                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.header.stamp = stamp
                    marker.ns = 'underground'
                    marker.id = marker_id
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD

                    marker.pose.position.x = x * self.cell_size + self.cell_size / 2
                    marker.pose.position.y = y * self.cell_size + self.cell_size / 2
                    marker.pose.position.z = z_offset
                    marker.pose.orientation.w = 1.0

                    marker.scale.x = self.cell_size * step * 0.9
                    marker.scale.y = self.cell_size * step * 0.9
                    marker.scale.z = 0.5

                    r, g, b, _ = self.MINERAL_COLORS[dominant_idx]['color']
                    marker.color.r = r
                    marker.color.g = g
                    marker.color.b = b
                    marker.color.a = 0.3  # Tres transparent

                    marker.lifetime = Duration(sec=0, nanosec=0)
                    markers.markers.append(marker)
                    marker_id += 1

        self.underground_pub.publish(markers)

    def publish_science_markers(self, stamp):
        """Publie les cibles scientifiques prioritaires"""
        if not hasattr(self, 'science_targets') or self.science_targets is None:
            return

        markers = MarkerArray()
        marker_id = 0

        # Trouver les meilleures cibles
        threshold = np.percentile(self.science_targets, 90)  # Top 10%

        targets = np.argwhere(self.science_targets > threshold)

        for y, x in targets[:50]:  # Limiter a 50 cibles
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = stamp
            marker.ns = 'science_targets'
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(int(x) * self.cell_size)
            marker.pose.position.y = float(int(y) * self.cell_size)
            marker.pose.position.z = 3.0
            marker.pose.orientation.w = 1.0

            priority = float(self.science_targets[y, x])
            size = 0.5 + priority * 1.5
            marker.scale.x = size
            marker.scale.y = size
            marker.scale.z = size

            # Couleur cyan pour les cibles
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 0.6

            marker.lifetime = Duration(sec=0, nanosec=0)
            markers.markers.append(marker)
            marker_id += 1

        self.science_markers_pub.publish(markers)

    def publish_reference_grid(self):
        """Publie une grille de reference"""
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        marker_id = 0

        # Lignes de la grille (tous les 10 unites)
        grid_step = 10

        for i in range(0, self.map_width + 1, grid_step):
            # Ligne verticale
            line = Marker()
            line.header.frame_id = 'map'
            line.header.stamp = stamp
            line.ns = 'grid'
            line.id = marker_id
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD

            line.pose.orientation.w = 1.0
            line.scale.x = 0.1

            line.color.r = 0.5
            line.color.g = 0.5
            line.color.b = 0.5
            line.color.a = 0.3

            p1 = Point()
            p1.x = float(i * self.cell_size)
            p1.y = 0.0
            p1.z = 0.01

            p2 = Point()
            p2.x = float(i * self.cell_size)
            p2.y = float(self.map_height * self.cell_size)
            p2.z = 0.01

            line.points = [p1, p2]
            line.lifetime = Duration(sec=0, nanosec=0)
            markers.markers.append(line)
            marker_id += 1

        for j in range(0, self.map_height + 1, grid_step):
            # Ligne horizontale
            line = Marker()
            line.header.frame_id = 'map'
            line.header.stamp = stamp
            line.ns = 'grid'
            line.id = marker_id
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD

            line.pose.orientation.w = 1.0
            line.scale.x = 0.1

            line.color.r = 0.5
            line.color.g = 0.5
            line.color.b = 0.5
            line.color.a = 0.3

            p1 = Point()
            p1.x = 0.0
            p1.y = float(j * self.cell_size)
            p1.z = 0.01

            p2 = Point()
            p2.x = float(self.map_width * self.cell_size)
            p2.y = float(j * self.cell_size)
            p2.z = 0.01

            line.points = [p1, p2]
            line.lifetime = Duration(sec=0, nanosec=0)
            markers.markers.append(line)
            marker_id += 1

        # Sol de base
        ground = Marker()
        ground.header.frame_id = 'map'
        ground.header.stamp = stamp
        ground.ns = 'ground'
        ground.id = 0
        ground.type = Marker.CUBE
        ground.action = Marker.ADD

        ground.pose.position.x = self.map_width * self.cell_size / 2
        ground.pose.position.y = self.map_height * self.cell_size / 2
        ground.pose.position.z = -0.1
        ground.pose.orientation.w = 1.0

        ground.scale.x = float(self.map_width * self.cell_size)
        ground.scale.y = float(self.map_height * self.cell_size)
        ground.scale.z = 0.1

        ground.color.r = 0.55   # Sable/argile naturel
        ground.color.g = 0.45
        ground.color.b = 0.30
        ground.color.a = 1.0

        ground.lifetime = Duration(sec=10, nanosec=0)
        markers.markers.append(ground)

        self.grid_pub.publish(markers)

    def publish_legend(self):
        """Publie une legende pour les types de mineraux"""
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        marker_id = 0

        # Position de la legende (en dehors de la carte)
        legend_x = -10.0
        legend_y = 0.0

        # Titre
        title = Marker()
        title.header.frame_id = 'map'
        title.header.stamp = stamp
        title.ns = 'legend'
        title.id = marker_id
        title.type = Marker.TEXT_VIEW_FACING
        title.action = Marker.ADD

        title.pose.position.x = legend_x
        title.pose.position.y = legend_y + 20.0
        title.pose.position.z = 5.0

        title.scale.z = 2.0
        title.color.r = 1.0
        title.color.g = 1.0
        title.color.b = 1.0
        title.color.a = 1.0

        title.text = "REE MINERALS"
        title.lifetime = Duration(sec=15, nanosec=0)
        markers.markers.append(title)
        marker_id += 1

        # Entrees de la legende
        for idx, info in self.MINERAL_COLORS.items():
            # Cube de couleur
            cube = Marker()
            cube.header.frame_id = 'map'
            cube.header.stamp = stamp
            cube.ns = 'legend_cube'
            cube.id = marker_id
            cube.type = Marker.SPHERE
            cube.action = Marker.ADD

            cube.pose.position.x = legend_x
            cube.pose.position.y = legend_y + 15.0 - idx * 5.0
            cube.pose.position.z = 2.0
            cube.pose.orientation.w = 1.0

            cube.scale.x = 1.8
            cube.scale.y = 1.8
            cube.scale.z = 1.8

            r, g, b, a = info['color']
            cube.color.r = r
            cube.color.g = g
            cube.color.b = b
            cube.color.a = a

            cube.lifetime = Duration(sec=15, nanosec=0)
            markers.markers.append(cube)
            marker_id += 1

            # Label
            label = Marker()
            label.header.frame_id = 'map'
            label.header.stamp = stamp
            label.ns = 'legend_text'
            label.id = marker_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD

            label.pose.position.x = legend_x + 5.0
            label.pose.position.y = legend_y + 15.0 - idx * 5.0
            label.pose.position.z = 2.0

            label.scale.z = 1.5
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.color.a = 1.0

            label.text = info['name']
            label.lifetime = Duration(sec=15, nanosec=0)
            markers.markers.append(label)
            marker_id += 1

        self.legend_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)

    try:
        node = REEVisualizationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in visualization node: {e}')
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
