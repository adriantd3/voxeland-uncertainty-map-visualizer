# General configuration
INPUT_PLY = "206_map.ply"

SCENE = 206
PRE_DIS_JSON_MAP = f"{SCENE}_pre_dis.json"
POST_DIS_JSON_MAP = f"{SCENE}_post_dis.json"

CAMERA_JSON = "camera.json"
VOXEL_SIZE = 0.04

# COLORS
HEATMAP_COLORS = [
        (0.00, "#000080"),  # 0% → navy
        (0.50, "#0000FF"),  # 25% → pure blue
        (0.75, "#00FFFF"),  # 50% → cyan
        (1.10, "#FFFF00"),  # 75% → yellow
        (1.5, "#FF0000"),  # 100% → red
]

