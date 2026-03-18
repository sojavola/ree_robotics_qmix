import numpy as np
from scipy.ndimage import gaussian_filter, sobel, label, binary_dilation


class AdvancedMineralGenerator:
    """
    Génère des cartes minérales REE réalistes :
    - Petits gisements épars (pas de grandes zones uniformes)
    - Chaque type REE a sa propre rareté et taille de gisement
    - Profil de concentration gaussien du centre vers les bords
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Propriétés géologiques de chaque type REE
        # n_deposits     : nombre de gisements sur la carte
        # radius_range   : (min, max) rayon en cellules
        # peak_conc      : concentration maximale au centre [0-1]
        # depth_bias     : 0=surface, 1=profond (pour couches souterraines)
        self.mineral_types = {
            # Oxydes REE — grands gisements primaires
            'REE_Oxides': {
                'color': [1.0, 0.4, 0.0],
                'n_deposits': 5,
                'radius_range': (7, 13),
                'peak_conc': 0.88,
                'depth_bias': 0.3,
            },
            # Silicates REE — gisements moyens, nombreux
            'REE_Silicates': {
                'color': [0.1, 0.75, 0.2],
                'n_deposits': 7,
                'radius_range': (5, 10),
                'peak_conc': 0.78,
                'depth_bias': 0.5,
            },
            # Phosphates REE — rares mais très concentrés
            'REE_Phosphates': {
                'color': [0.55, 0.1, 0.8],
                'n_deposits': 4,
                'radius_range': (5, 9),
                'peak_conc': 0.92,
                'depth_bias': 0.2,
            },
            # Carbonates REE — diffus, répartis sur tout le terrain
            'REE_Carbonates': {
                'color': [0.9, 0.75, 0.1],
                'n_deposits': 6,
                'radius_range': (5, 11),
                'peak_conc': 0.72,
                'depth_bias': 0.4,
            },
        }

    # ──────────────────────────────────────────────────────────────────
    def generate_geological_map(self, seed=0):
        """Génère la carte minérale. Même seed = même carte."""
        np.random.seed(seed)
        mineral_map = np.zeros((self.height, self.width, len(self.mineral_types)))

        for idx, (name, props) in enumerate(self.mineral_types.items()):
            mineral_map[:, :, idx] = self._place_deposits(props)

        return mineral_map

    # ──────────────────────────────────────────────────────────────────
    def _place_deposits(self, props):
        """
        Place N gisements avec distribution zonée sur toute la carte.
        La carte est divisée en zones pour garantir une dispersion uniforme.
        """
        layer = np.zeros((self.height, self.width), dtype=np.float32)

        n            = props['n_deposits']
        r_min, r_max = props['radius_range']
        peak         = props['peak_conc']

        ys = np.arange(self.height)
        xs = np.arange(self.width)
        xx, yy = np.meshgrid(xs, ys)
        margin = r_max + 3

        # Grille de zones pour forcer la dispersion spatiale
        n_zones_x = max(2, int(np.ceil(np.sqrt(n * self.width / self.height))))
        n_zones_y = max(2, int(np.ceil(n / n_zones_x)) + 1)
        zone_w = (self.width  - 2 * margin) / n_zones_x
        zone_h = (self.height - 2 * margin) / n_zones_y

        all_zones = [(zi, zj) for zi in range(n_zones_x) for zj in range(n_zones_y)]
        np.random.shuffle(all_zones)
        chosen_zones = all_zones[:n]

        for zi, zj in chosen_zones:
            x_lo = int(margin + zi * zone_w)
            x_hi = int(margin + (zi + 1) * zone_w)
            y_lo = int(margin + zj * zone_h)
            y_hi = int(margin + (zj + 1) * zone_h)

            cx = np.random.randint(max(margin, x_lo), min(self.width  - margin, x_hi) + 1)
            cy = np.random.randint(max(margin, y_lo), min(self.height - margin, y_hi) + 1)
            r  = np.random.uniform(r_min, r_max)

            # Ellipse peu allongée → clusters circulaires, pas de lignes
            angle = np.random.uniform(0, np.pi)
            ratio = np.random.uniform(0.60, 0.95)

            dx = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
            dy = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
            dist2 = dx ** 2 + (dy / ratio) ** 2

            peak_actual = peak * np.random.uniform(0.70, 1.0)
            sigma = r / 2.5
            gaussian = peak_actual * np.exp(-dist2 / (2 * sigma ** 2))
            mask = dist2 <= (r * 1.1) ** 2

            noise = 1.0 + np.random.uniform(-0.06, 0.06, size=(self.height, self.width))
            layer = np.maximum(layer, gaussian * noise * mask)

        return np.clip(layer, 0.0, 1.0)

    # ──────────────────────────────────────────────────────────────────
    def generate_underground_layers(self, surface_mineral_map, num_layers=3):
        """Couches souterraines avec gradient selon depth_bias."""
        underground_layers = []
        for depth in range(num_layers):
            depth_factor = (depth + 1) / num_layers
            layer_map = surface_mineral_map.copy()
            for midx, (_, props) in enumerate(self.mineral_types.items()):
                db = props['depth_bias']
                if db < 0.5:
                    reduction = (1.0 - db) * depth_factor
                    layer_map[:, :, midx] *= (1.0 - reduction)
                else:
                    enhancement = (db - 0.5) * depth_factor * 2
                    layer_map[:, :, midx] *= (1.0 + enhancement)
            underground_layers.append(np.clip(layer_map, 0, 1))
        return underground_layers

    # ──────────────────────────────────────────────────────────────────
    def detect_mineral_clusters(self, mineral_map, mineral_idx,
                                 min_samples=3, eps=2.5):
        """Détecte les gisements via composantes connexes."""
        layer = mineral_map[:, :, mineral_idx]
        mask  = layer > 0.15

        if np.count_nonzero(mask) < min_samples:
            return []

        r = int(np.ceil(eps))
        cy, cx = np.ogrid[-r:r + 1, -r:r + 1]
        struct   = (cy ** 2 + cx ** 2) <= eps ** 2
        dilated  = binary_dilation(mask, structure=struct)
        labeled_map, num_features = label(dilated)

        clusters = []
        for cid in range(1, num_features + 1):
            pts = np.argwhere(mask & (labeled_map == cid))
            if len(pts) < min_samples:
                continue
            clusters.append({
                'points': pts,
                'center': np.mean(pts, axis=0),
                'size': len(pts),
                'max_concentration': float(
                    np.max(layer[pts[:, 0], pts[:, 1]])
                ),
            })
        return clusters
