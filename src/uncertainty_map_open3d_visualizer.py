from typing import Dict
from datetime import datetime
import numpy as np
import open3d as o3d
import matplotlib.colors as mcolors
import config

from metrics.max_entropy_from_ply import get_max_property_value
from utils.map_json_reader import get_instances_entropy

def load_point_cloud(file_path: str) -> o3d.t.geometry.PointCloud:
    """Load a point cloud (Tensor API) from a PLY file.
    Args:
        file_path: Path to the input PLY file.
    Returns:
        A Tensor-based PointCloud containing all attributes.
    """
    pcd_t = o3d.t.io.read_point_cloud(file_path)
    if np.asarray(pcd_t.point.positions.shape)[0] == 0:
        raise RuntimeError(f"Failed to load point cloud from '{file_path}' or it is empty.")
    return pcd_t


def adjust_uncertainty_for_instances(pcd_t: o3d.t.geometry.PointCloud) -> None:
    """Overwrite uncertainty_categories to its minimum value where instanceid == 0.
    Modifies the PointCloud in place.
    Args:
        pcd_t: A Tensor-based PointCloud that must contain the attributes
               "uncertainty_categories" and "instanceid".
    """
    uncert_tensor = pcd_t.point["uncertainty_categories"]
    instanceid_tensor = pcd_t.point["instanceid"]

    # Compute the minimum uncertainty across all points
    uncert_np = uncert_tensor.numpy().reshape(-1)
    umin_value = float(np.min(uncert_np))
    umin_tensor = o3d.core.Tensor(umin_value, dtype=uncert_tensor.dtype)

    # Build a boolean mask where instanceid == 0
    mask_instance_zero = instanceid_tensor == 0

    # Overwrite uncertainty to umin where instanceid == 0
    uncert_tensor[mask_instance_zero] = umin_tensor
    pcd_t.point["uncertainty_categories"] = uncert_tensor


def compute_heatmap_colors(
    pcd_t: o3d.t.geometry.PointCloud,
    instances_entropy: Dict[int, float] = None
) -> np.ndarray:
    # 1) Sacar IDs de instancia
    if 'instanceid' not in pcd_t.point:
        raise KeyError("No hay atributo 'instanceid' en el PointCloud")
    inst_ids = np.asarray(
        pcd_t.point['instanceid'].cpu().numpy()
    ).flatten().astype(int)

    # 2) Construir array de entropías “crudas”
    if instances_entropy is not None:
        entropies = np.array(
            [instances_entropy.get(i, 0.75) for i in inst_ids],
            dtype=float
        )
    else:
        entropies = np.zeros_like(inst_ids, dtype=float)

    # 3) Normalizar solo las claves de HEATMAP_COLORS al rango [0,1]
    max_key = config.HEATMAP_COLORS[-1][0]  # p.ej. 1.5
    cmap_list = [
        (key / max_key, color)
        for key, color in config.HEATMAP_COLORS
    ]
    heatmap_cmap = mcolors.LinearSegmentedColormap.from_list(
        "heatmap_cmap", cmap_list
    )

    # 4) Aplicar Normalize con tus umbrales fijos y clip=True
    norm = mcolors.Normalize(vmin=0.0, vmax=max_key, clip=True)
    colors_rgba = heatmap_cmap(norm(entropies))
    colors_rgb = colors_rgba[:, :3]

    return colors_rgb.astype(np.float32)

def assign_colors_to_point_cloud(
    pcd_t: o3d.t.geometry.PointCloud, colors: np.ndarray
) -> None:
    """Assign an RGB color array to the PointCloud.

    Modifies the PointCloud in place by creating or overwriting the "colors" attribute.

    Args:
        pcd_t: A Tensor-based PointCloud.
        colors: A NumPy array of shape (N, 3) with RGB values in [0, 1].
    """
    color_tensor = o3d.core.Tensor(colors, dtype=o3d.core.Dtype.Float32)
    pcd_t.point["colors"] = color_tensor


def create_voxel_grid(
    pcd_t: o3d.t.geometry.PointCloud, voxel_size: float
) -> o3d.geometry.VoxelGrid:
    """Convert a Tensor PointCloud to legacy format and create a VoxelGrid.

    Args:
        pcd_t: A Tensor-based PointCloud with a "colors" attribute.
        voxel_size: The side length of each voxel.

    Returns:
        A VoxelGrid object ready for visualization.
    """
    pcd_legacy = o3d.t.geometry.PointCloud.to_legacy(pcd_t)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_legacy, voxel_size=voxel_size)
    return voxel_grid


def load_camera_parameters(json_path: str) -> o3d.camera.PinholeCameraParameters:
    """Read PinholeCameraParameters from a JSON file.
    Args:
        json_path: Path to the JSON file containing camera parameters.
    Returns:
        A PinholeCameraParameters object.
    """
    return o3d.io.read_pinhole_camera_parameters(json_path)

def save_screenshot_and_camera(vis):

    image = vis.capture_screen_float_buffer(False)
    img = (np.asarray(image) * 255).astype(np.uint8)
    image_name = f"screenshot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    o3d.io.write_image(f"saves/{image_name}", o3d.geometry.Image(img))

    cam_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters("saves/camera_params.json", cam_params)

    print(f"Saved {image_name} and camera_params.json")

    return False

def visualize_voxel_grid(
    voxel_grid: o3d.geometry.VoxelGrid,
    camera_params: o3d.camera.PinholeCameraParameters = None,
    window_name: str = "Voxel Visualization",
    width: int = 1280,
    height: int = 720,
) -> None:
    """Open a Visualizer window and display the VoxelGrid with an optional camera pose.
    Args:
        voxel_grid: The VoxelGrid to visualize.
        camera_params: (Optional) PinholeCameraParameters for initial camera pose.
        window_name: Title of the visualization window.
        width: Window width in pixels.
        height: Window height in pixels.
    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name, width=width, height=height)
    vis.add_geometry(voxel_grid)
    vis.register_key_callback(ord('S'), lambda vis: save_screenshot_and_camera(vis))

    if camera_params is not None:
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(camera_params)

    vis.run()
    vis.destroy_window()


def main():

    # 1. Load and adjust uncertainties for instanceid == 0
    pcd_t = load_point_cloud(f"ply_maps/{config.INPUT_PLY}")
    adjust_uncertainty_for_instances(pcd_t)

    # 2. Compute RGB colors based on updated uncertainty
    entropy_dict = get_instances_entropy(json_path=f"json_map/{config.POST_DIS_JSON_MAP}")

    colors = compute_heatmap_colors(pcd_t, entropy_dict)
    assign_colors_to_point_cloud(pcd_t, colors)

    # 3. Create VoxelGrid from the colored point cloud
    voxel_grid = create_voxel_grid(pcd_t, voxel_size=config.VOXEL_SIZE)

    # 4. Load camera parameters (optional)
    try:
        camera_params = load_camera_parameters(f"saves/{config.CAMERA_JSON}")
    except Exception:
        camera_params = None
    # 5. Visualize the result
    visualize_voxel_grid(
        voxel_grid,
        camera_params=camera_params,
        window_name="Voxeland Uncertainty Scene Map",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()