import numpy as np
import open3d as o3d
import matplotlib.colors as mcolors
import config


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
    pcd_t: o3d.t.geometry.PointCloud
) -> np.ndarray:
    """Compute RGB colors from normalized uncertainty_categories using a custom heatmap.
    Args:
        pcd_t: A Tensor-based PointCloud with an updated
               "uncertainty_categories" attribute.
    Returns:
        A NumPy array of shape (N, 3) with RGB values in [0, 1].
    """
    uncert_np = pcd_t.point["uncertainty_categories"].numpy().reshape(-1).astype(np.float32)
    umin, umax = uncert_np.min(), uncert_np.max()

    if umin == umax:
        # If all values are equal, use a mid-level gray (0.5) for normalization
        norm_uncert = np.full_like(uncert_np, 0.5, dtype=np.float32)
    else:
        norm_uncert = (uncert_np - umin) / (umax - umin)

    # Define a custom colormap: dark blue → red
    heatmap_cmap = mcolors.LinearSegmentedColormap.from_list("heatmap_cmap", config.HEATMAP_COLORS)

    # Map normalized uncertainty to RGB (discard alpha channel)
    colors_rgba = heatmap_cmap(norm_uncert)
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
    o3d.io.write_image("../test_demo_image.png", o3d.geometry.Image(img))

    cam_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters("camera_params.json", cam_params)

    print("Saved test_demo_image.png and camera_params.json")

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
    pcd_t = load_point_cloud(f"maps/{config.INPUT_PLY}")
    adjust_uncertainty_for_instances(pcd_t)

    # 2. Compute RGB colors based on updated uncertainty
    colors = compute_heatmap_colors(pcd_t)
    assign_colors_to_point_cloud(pcd_t, colors)

    # 3. Create VoxelGrid from the colored point cloud
    voxel_grid = create_voxel_grid(pcd_t, voxel_size=config.VOXEL_SIZE)

    # 4. Load camera parameters (optional)
    try:
        camera_params = load_camera_parameters(f"{config.CAMERA_JSON}")
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