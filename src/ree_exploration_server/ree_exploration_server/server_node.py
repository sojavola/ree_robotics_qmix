
#!/usr/bin/env python3
# server_node.py - Version avec générateur avancé

import rclpy
from rclpy.node import Node
import numpy as np
import threading
import time
from scipy.ndimage import gaussian_filter

from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import OccupancyGrid

# === IMPORT DES GÉNÉRATEURS ===
try:
    from .advanced_mineral_generator import AdvancedMineralGenerator
    ADVANCED_GENERATOR_AVAILABLE = True
    print("✅ AdvancedMineralGenerator importé avec succès")
except ImportError as e:
    ADVANCED_GENERATOR_AVAILABLE = False
    print(f"⚠️ AdvancedMineralGenerator non disponible: {e}")

# Générateur basique (fallback)
class MineralMapGenerator:
    def __init__(self, width=100, height=100):
        self.width = width
        self.height = height
        
    def generate_mineral_map(self):
        """Génère une carte minérale procédurale basique"""
        mineral_map = np.zeros((self.height, self.width, 4))
        
        for mineral_idx in range(4):
            base = np.random.rand(self.height, self.width)
            filtered = gaussian_filter(base, sigma=3.0)
            threshold = 0.6 + np.random.random() * 0.3
            deposits = np.where(filtered > threshold, filtered, 0)
            
            if np.max(deposits) > 0:
                deposits = deposits / np.max(deposits)
            
            mineral_map[:, :, mineral_idx] = deposits
        
        return mineral_map

class REEExplorationServer(Node):
    def __init__(self):
        super().__init__('ree_exploration_server')
        
        # === CONFIGURATION ===
        self.map_width = 100
        self.map_height = 100
        self.num_robots = 4
        
        # === CHOIX DU GÉNÉRATEUR (configurable) ===
        self.declare_parameter('use_advanced_generator', True)
        use_advanced = self.get_parameter('use_advanced_generator').get_parameter_value().bool_value
        
        if use_advanced and ADVANCED_GENERATOR_AVAILABLE:
            self.get_logger().info('🧠 Utilisation du générateur minéral AVANCÉ')
            self.mineral_generator = AdvancedMineralGenerator(self.map_width, self.map_height)
            self.generator_type = 'advanced'
        else:
            self.get_logger().info('🧠 Utilisation du générateur minéral BASIC')
            self.mineral_generator = MineralMapGenerator(self.map_width, self.map_height)
            self.generator_type = 'basic'
        
        # === ÉTAT DU SYSTÈME ===
        self.mineral_map = None
        self.underground_layers = []  # Pour le générateur avancé
        self.robot_positions = {}
        self.exploration_map = np.zeros((self.map_height, self.map_width))
        self.obstacle_map = np.zeros((self.map_height, self.map_width))
        
        # === VERROU THREAD-SAFE ===
        self.lock = threading.Lock()
        
        # === PUBLISHERS ===
        self.mineral_pub = self.create_publisher(Float32MultiArray, '/mineral_map', 10)
        self.obstacle_pub = self.create_publisher(OccupancyGrid, '/obstacle_map', 10)
        self.science_pub = self.create_publisher(Float32MultiArray, '/science_targets', 10)
        self.status_pub = self.create_publisher(String, '/system_status', 10)
        
        # Publier aussi les couches souterraines si disponibles
        if self.generator_type == 'advanced':
            self.underground_pub = self.create_publisher(
                Float32MultiArray, 
                '/underground_layers', 
                10
            )
        
        # === SUBSCRIBERS ===
        for i in range(self.num_robots):
            # Positions des robots
            self.create_subscription(
                Pose2D,
                f'/robot_{i}/position',
                self.create_position_callback(i),
                10
            )
            
            # Actions de nettoyage
            self.create_subscription(
                Float32MultiArray,
                f'/robot_{i}/cleaning_action',
                self.create_cleaning_callback(i),
                10
            )
        
        # === TIMERS ===
        self.map_timer = self.create_timer(2.0, self.publish_maps)
        self.status_timer = self.create_timer(5.0, self.publish_status)
        self.update_timer = self.create_timer(1.0, self.update_system)
        
        # Timer pour les couches souterraines (si avancé)
        if self.generator_type == 'advanced':
            self.underground_timer = self.create_timer(5.0, self.publish_underground_layers)
        
        # === INITIALISATION ===
        self.initialize_system()
        
        self.get_logger().info('🚀 REE Exploration Server Node initialized')
        self.get_logger().info(f'📊 Map: {self.map_width}x{self.map_height}')
        self.get_logger().info(f'🤖 Robots: {self.num_robots}')
        self.get_logger().info(f'⚙️ Generator: {self.generator_type.upper()}')
    
    def create_position_callback(self, robot_id):
        """Callback pour position du robot"""
        def callback(msg):
            with self.lock:
                self.robot_positions[robot_id] = (msg.x, msg.y)
                self.update_exploration_map(msg.x, msg.y)
                self.get_logger().debug(f'🤖 Robot {robot_id}: ({msg.x:.1f}, {msg.y:.1f})')
        return callback
    
    def create_cleaning_callback(self, robot_id):
        """Callback pour nettoyage"""
        def callback(msg):
            if len(msg.data) >= 2:
                x, y = int(msg.data[0]), int(msg.data[1])
                self.clean_area(x, y)
                # Optionnel: log détaillé
                # self.get_logger().info(f'🧹 Robot {robot_id} cleaned: ({x}, {y})')
        return callback
    
    def initialize_system(self):
        """Initialise le système avec le générateur approprié"""
        with self.lock:
            # === GÉNÉRATION DE LA CARTE ===
            if self.generator_type == 'advanced':
                # Méthode avancée
                self.mineral_map = self.mineral_generator.generate_geological_map()
                
                # Générer couches souterraines
                self.underground_layers = self.mineral_generator.generate_underground_layers(
                    self.mineral_map, 
                    num_layers=3
                )
                
                self.get_logger().info('✅ Carte géologique avancée générée')
                self.get_logger().info(f'   Types REE: 4 types réalistes')
                self.get_logger().info(f'   Couches: {len(self.underground_layers)} souterraines')
                
                # Analyser les clusters pour debug
                self.analyze_mineral_clusters()
                
            else:
                # Méthode basique
                self.mineral_map = self.mineral_generator.generate_mineral_map()
                self.get_logger().info('✅ Carte minérale basique générée')
            
            # === GÉNÉRATION OBSTACLES ===
            self.generate_obstacles()
            
            # === POSITIONS INITIALES ROBOTS ===
            for i in range(self.num_robots):
                self.robot_positions[i] = self.get_valid_start_position()
        
        self.get_logger().info('✅ System initialized')
    
    def analyze_mineral_clusters(self):
        """Analyse les clusters minéraux (debug)"""
        if self.generator_type != 'advanced':
            return
        
        mineral_names = ['REE_Oxides', 'REE_Silicates', 
                        'REE_Phosphates', 'REE_Carbonates']
        
        total_clusters = 0
        for mineral_idx, mineral_name in enumerate(mineral_names):
            clusters = self.mineral_generator.detect_mineral_clusters(
                self.mineral_map, 
                mineral_idx,
                min_samples=5,
                eps=2.5
            )
            
            if clusters:
                total_clusters += len(clusters)
                self.get_logger().info(
                    f'📊 {mineral_name}: {len(clusters)} clusters'
                )
        
        self.get_logger().info(f'📈 Total clusters détectés: {total_clusters}')
        
        # Calculer les statistiques globales
        max_concentration = np.max(self.mineral_map)
        avg_concentration = np.mean(self.mineral_map[self.mineral_map > 0.1])
        coverage = np.sum(self.mineral_map > 0.1) / self.mineral_map.size * 100
        
        self.get_logger().info(
            f'📈 Stats: Max={max_concentration:.2f}, '
            f'Avg={avg_concentration:.2f}, Coverage={coverage:.1f}%'
        )
    
    def generate_obstacles(self):
        """Génère des obstacles réalistes"""
        self.obstacle_map = np.zeros((self.map_height, self.map_width))
        
        # Bords
        self.obstacle_map[0, :] = 100
        self.obstacle_map[-1, :] = 100
        self.obstacle_map[:, 0] = 100
        self.obstacle_map[:, -1] = 100
        
        # Obstacles internes (adaptés à la géologie si avancé)
        if self.generator_type == 'advanced':
            # Utiliser la carte géologique pour placer les obstacles
            geology = np.mean(self.mineral_map, axis=2)
            
            # Les zones de forte concentration peuvent être rocheuses
            rocky_areas = geology > 0.7
            self.obstacle_map[rocky_areas] = 100
            
            # Ajouter quelques obstacles aléatoires dans les zones neutres
            num_obstacles = 8
            for _ in range(num_obstacles):
                x = np.random.randint(5, self.map_width - 5)
                y = np.random.randint(5, self.map_height - 5)
                
                # Éviter les zones minérales importantes
                if geology[y, x] < 0.3:
                    size = np.random.randint(2, 6)
                    self.add_circular_obstacle(x, y, size)
        else:
            # Méthode basique
            num_obstacles = 10
            for _ in range(num_obstacles):
                x = np.random.randint(10, self.map_width - 10)
                y = np.random.randint(10, self.map_height - 10)
                size = np.random.randint(3, 8)
                self.add_circular_obstacle(x, y, size)
    
    def add_circular_obstacle(self, x, y, radius):
        """Ajoute un obstacle circulaire"""
        for i in range(max(0, y-radius), min(self.map_height, y+radius+1)):
            for j in range(max(0, x-radius), min(self.map_width, x+radius+1)):
                if np.sqrt((i-y)**2 + (j-x)**2) <= radius:
                    self.obstacle_map[i, j] = 100
    
    def get_valid_start_position(self):
        """Retourne une position de départ valide"""
        while True:
            x = np.random.randint(10, self.map_width - 10)
            y = np.random.randint(10, self.map_height - 10)
            
            if self.obstacle_map[y, x] == 0:
                # Éviter les zones minérales très denses pour commencer
                if self.generator_type == 'advanced':
                    if np.max(self.mineral_map[y, x, :]) < 0.8:
                        return (x, y)
                else:
                    return (x, y)
    
    def update_exploration_map(self, x, y):
        """Met à jour la carte d'exploration"""
        exploration_radius = 5
        x_int = int(x)
        y_int = int(y)
        
        for i in range(max(0, y_int-exploration_radius), min(self.map_height, y_int+exploration_radius+1)):
            for j in range(max(0, x_int-exploration_radius), min(self.map_width, x_int+exploration_radius+1)):
                distance = np.sqrt((i-y_int)**2 + (j-x_int)**2)
                if distance <= exploration_radius:
                    self.exploration_map[i, j] = min(1.0, self.exploration_map[i, j] + 0.3)
    
    def clean_area(self, x, y):
        """Détecte les minéraux à la position du robot — carte statique, rien n'est détruit"""
        x_int = int(x)
        y_int = int(y)
        if not (0 <= y_int < self.map_height and 0 <= x_int < self.map_width):
            return
        with self.lock:
            mineral_at_pos = self.mineral_map[y_int, x_int, :]
        if np.max(mineral_at_pos) > 0.3:
            self.get_logger().debug(
                f'🔍 Mineral detected at ({x_int}, {y_int}): '
                f'max={np.max(mineral_at_pos):.2f}'
            )
    
    def update_system(self):
        """Carte statique — les minéraux REE ne bougent pas entre les épisodes"""
        pass
    
    def publish_maps(self):
        """Publie les cartes mises à jour"""
        with self.lock:
            # === CARTE MINÉRALE ===
            mineral_msg = Float32MultiArray()
            mineral_msg.layout.dim.append(self.create_multi_array_dimension(
                "height", self.map_height, self.map_height * self.map_width * 4))
            mineral_msg.layout.dim.append(self.create_multi_array_dimension(
                "width", self.map_width, self.map_width * 4))
            mineral_msg.layout.dim.append(self.create_multi_array_dimension(
                "channels", 4, 4))
            
            mineral_msg.data = self.mineral_map.flatten().tolist()
            self.mineral_pub.publish(mineral_msg)
            
            # === CARTE OBSTACLES ===
            obstacle_msg = OccupancyGrid()
            obstacle_msg.header.stamp = self.get_clock().now().to_msg()
            obstacle_msg.header.frame_id = "map"
            obstacle_msg.info.width = self.map_width
            obstacle_msg.info.height = self.map_height
            obstacle_msg.info.resolution = 0.1
            obstacle_msg.data = self.obstacle_map.flatten().astype(np.int8).tolist()
            self.obstacle_pub.publish(obstacle_msg)
            
            # === CIBLES SCIENTIFIQUES ===
            science_msg = Float32MultiArray()
            science_targets = self.calculate_science_targets()
            
            science_msg.layout.dim.append(self.create_multi_array_dimension(
                "height", self.map_height, self.map_height * self.map_width))
            science_msg.layout.dim.append(self.create_multi_array_dimension(
                "width", self.map_width, self.map_width))
            
            science_msg.data = science_targets.flatten().tolist()
            self.science_pub.publish(science_msg)
        
        # Log périodique
        if hasattr(self, 'map_publish_count'):
            self.map_publish_count += 1
        else:
            self.map_publish_count = 0
            
        if self.map_publish_count % 5 == 0:
            self.get_logger().debug('🗺️ Maps published')
    
    def publish_underground_layers(self):
        """Publie les couches souterraines (si disponible)"""
        if self.generator_type != 'advanced' or not self.underground_layers:
            return
        
        with self.lock:
            # Publier la première couche souterraine (profondeur 1)
            underground_msg = Float32MultiArray()
            layer = self.underground_layers[0]  # Première couche
            
            underground_msg.layout.dim.append(self.create_multi_array_dimension(
                "height", self.map_height, self.map_height * self.map_width * 4))
            underground_msg.layout.dim.append(self.create_multi_array_dimension(
                "width", self.map_width, self.map_width * 4))
            underground_msg.layout.dim.append(self.create_multi_array_dimension(
                "channels", 4, 4))
            
            underground_msg.data = layer.flatten().tolist()
            self.underground_pub.publish(underground_msg)
            
            # Log
            if hasattr(self, 'underground_publish_count'):
                self.underground_publish_count += 1
            else:
                self.underground_publish_count = 0
                
            if self.underground_publish_count % 10 == 0:
                self.get_logger().debug('⛰️ Underground layer published')
    
    def create_multi_array_dimension(self, label, size, stride):
        """Crée une dimension pour MultiArray message"""
        from std_msgs.msg import MultiArrayDimension
        dim = MultiArrayDimension()
        dim.label = label
        dim.size = size
        dim.stride = stride
        return dim
    
    def calculate_science_targets(self):
        """Calcule les cibles scientifiques prioritaires"""
        unexplored = 1.0 - self.exploration_map
        
        if self.generator_type == 'advanced':
            # Pour le générateur avancé: prioriser les zones de fort gradient
            mineral_gradient = np.zeros((self.map_height, self.map_width))
            for i in range(4):
                layer = self.mineral_map[:, :, i]
                gradient = np.abs(np.gradient(layer)[0]) + np.abs(np.gradient(layer)[1])
                mineral_gradient += gradient
            
            # Cibles = zones inexplorées avec fort gradient géologique
            targets = unexplored * (mineral_gradient / 4.0)
        else:
            # Pour le générateur basique
            mineral_potential = np.max(self.mineral_map, axis=2)
            targets = unexplored * mineral_potential
        
        return targets
    
    
    def publish_status(self):
        """Publie le statut du système"""
        with self.lock:
            coverage = np.mean(self.exploration_map) * 100
            
            if self.generator_type == 'advanced':
                mineral_stats = f"REE: {np.sum(self.mineral_map > 0.3):,} cells"
                clusters_info = f"Clusters: {self.analyze_cluster_count()}"
            else:
                mineral_density = np.mean(self.mineral_map) * 100
                mineral_stats = f"Density: {mineral_density:.1f}%"
                clusters_info = ""
            
            active_robots = len(self.robot_positions)
            
            status_text = (f"System | Robots: {active_robots} | "
                          f"Coverage: {coverage:.1f}% | "
                          f"{mineral_stats}")
            
            if clusters_info:
                status_text += f" | {clusters_info}"
        
        status_msg = String()
        status_msg.data = status_text
        self.status_pub.publish(status_msg)
        
        # Log périodique
        if hasattr(self, 'status_counter'):
            self.status_counter += 1
        else:
            self.status_counter = 0
            
        if self.status_counter % 6 == 0:  # Toutes les 30 secondes environ
            self.get_logger().info(f'📊 {status_text}')
    
    def analyze_cluster_count(self):
        """Compte rapide des clusters"""
        if self.generator_type != 'advanced':
            return "N/A"
        
        total_clusters = 0
        for mineral_idx in range(4):
            clusters = self.mineral_generator.detect_mineral_clusters(
                self.mineral_map, mineral_idx, min_samples=3, eps=2.0)
            total_clusters += len(clusters)
        
        return str(total_clusters)
    

def main():
    rclpy.init()
    
    try:
        node = REEExplorationServer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Shutting down server...')
    except Exception as e:
        node.get_logger().error(f'❌ Error in server node: {e}')
        import traceback
        traceback.print_exc()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()