import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json
INPUT_MESH_PATH = "your_mesh_file.obj"  # Replace with your mesh file path
OUTPUT_DIR = "output_meshes"
N_BINS = 1024
os.makedirs(OUTPUT_DIR, exist_ok=True)
def load_mesh(file_path):
    """Load mesh and return vertices, faces, and mesh object"""
    try:
        mesh = trimesh.load(file_path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"Error: Loaded object is not a valid Trimesh")
            return None, None, None
        vertices = mesh.vertices
        faces = mesh.faces
        return mesh, vertices, faces
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return None, None, None
def minmax_normalize(vertices):
    """
    Min-Max normalization: scales coordinates to [0, 1] range
    Formula: x' = (x - x_min) / (x_max - x_min)
    """
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    # Add epsilon to avoid division by zero
    normalized = (vertices - v_min) / (v_max - v_min + 1e-8)
    return normalized, v_min, v_max
def unit_sphere_normalize(vertices):
    """
    Unit Sphere normalization: centers at origin and scales to unit sphere
    - Compute centroid (mean of all vertices)
    - Center vertices at origin
    - Scale by max distance to fit in unit sphere
    """
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    max_distance = np.max(np.linalg.norm(centered, axis=1))
    normalized = centered / (max_distance + 1e-8)
    return normalized, centroid, max_distance

def quantize_vertices(normalized_vertices, n_bins):
    """
    Quantize normalized vertices to discrete bins
    Formula: q = floor(x' * (n_bins - 1))
    """
    quantized = np.floor(normalized_vertices * (n_bins - 1)).astype(np.int32)
    quantized = np.clip(quantized, 0, n_bins - 1)  # Ensure within valid range
    return quantized

def save_mesh(vertices, faces, output_path):
    """Save mesh to file (.ply or .obj format)"""
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    new_mesh.export(output_path)
    print(f"✓ Saved mesh to: {output_path}")

def visualize_mesh(vertices, title, ax=None, color_map='viridis'):
    """Visualize mesh vertices in 3D scatter plot"""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Color points by Z coordinate for better visualization
    scatter = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                        c=vertices[:, 2], cmap=color_map, s=1, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    return ax, scatter

def print_statistics(vertices, label="Vertices"):
    """Print vertex statistics"""
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    v_mean = vertices.mean(axis=0)
    v_std = vertices.std(axis=0)
    v_range = v_max - v_min
    
    print(f"
{label} Statistics:")
    print(f"  Min (X, Y, Z):       {v_min}")
    print(f"  Max (X, Y, Z):       {v_max}")
    print(f"  Range (X, Y, Z):     {v_range}")
    print(f"  Mean (X, Y, Z):      {v_mean}")
    print(f"  Std Dev (X, Y, Z):   {v_std}")

def main():
    print("="*80)
    print("TASK 2: NORMALIZE & QUANTIZE THE MESH")
    print("="*80)
    print("
[STEP 1] Loading mesh from file...")
    mesh, original_vertices, faces = load_mesh(INPUT_MESH_PATH)
    if mesh is None:
        print("Failed to load mesh. Exiting.")
        return
    print(f"✓ Successfully loaded mesh")
    print(f"  - Number of vertices: {original_vertices.shape[0]}")
    print(f"  - Number of faces: {faces.shape[0]}")
    print_statistics(original_vertices, "Original Mesh")
    print("
" + "-"*80)
    print("[STEP 2] Applying MIN-MAX NORMALIZATION...")
    print("-"*80)
    minmax_normalized, v_min, v_max = minmax_normalize(original_vertices)
    print(f"✓ Min-Max normalization completed")
    print(f"  - Original range: X=[{original_vertices[:, 0].min():.4f}, {original_vertices[:, 0].max():.4f}]")
    print(f"  - Normalized range: [{minmax_normalized.min():.4f}, {minmax_normalized.max():.4f}]")
    print_statistics(minmax_normalized, "Min-Max Normalized")
    save_mesh(minmax_normalized, faces, 
              os.path.join(OUTPUT_DIR, "01_minmax_normalized.ply"))
    print("
[STEP 3] Quantizing MIN-MAX normalized mesh...")
    print(f"  - Number of bins: {N_BINS}")
    print(f"  - Bit depth: {np.log2(N_BINS):.1f} bits per axis")
    
    minmax_quantized = quantize_vertices(minmax_normalized, N_BINS)
    
    print(f"✓ Quantization completed")
    print(f"  - Quantized range: [{minmax_quantized.min()}, {minmax_quantized.max()}]")
    minmax_quantized_float = minmax_quantized.astype(np.float64) / (N_BINS - 1)
    save_mesh(minmax_quantized_float, faces, 
              os.path.join(OUTPUT_DIR, "02_minmax_quantized.ply"))
    print_statistics(minmax_quantized_float, "Min-Max Quantized")
    print("
" + "-"*80)
    print("[STEP 4] Applying UNIT SPHERE NORMALIZATION...")
    print("-"*80)
    unitsphere_normalized, centroid, max_dist = unit_sphere_normalize(original_vertices)
    print(f"✓ Unit Sphere normalization completed")
    print(f"  - Mesh centroid: {centroid}")
    print(f"  - Max distance from centroid: {max_dist:.4f}")
    print(f"  - Normalized range: [{unitsphere_normalized.min():.4f}, {unitsphere_normalized.max():.4f}]")
    print_statistics(unitsphere_normalized, "Unit Sphere Normalized")
    save_mesh(unitsphere_normalized, faces, 
              os.path.join(OUTPUT_DIR, "03_unitsphere_normalized.ply"))
    print("
[STEP 5] Quantizing UNIT SPHERE normalized mesh...")
    unitsphere_shifted = (unitsphere_normalized + 1) / 2
    unitsphere_quantized = quantize_vertices(unitsphere_shifted, N_BINS)
    print(f"✓ Quantization completed")
    print(f"  - Quantized range: [{unitsphere_quantized.min()}, {unitsphere_quantized.max()}]")
    
    unitsphere_quantized_float = unitsphere_quantized.astype(np.float64) / (N_BINS - 1)
    save_mesh(unitsphere_quantized_float, faces, 
              os.path.join(OUTPUT_DIR, "04_unitsphere_quantized.ply"))
    
    print_statistics(unitsphere_quantized_float, "Unit Sphere Quantized")
    print("
" + "-"*80)
    print("[STEP 6] Creating visualizations...")
    print("-"*80)
    
    fig = plt.figure(figsize=(20, 12))
    print("  - Plotting Original Mesh...")
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    visualize_mesh(original_vertices, "1. Original Mesh", ax1)
    print("  - Plotting Min-Max Normalized...")
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    visualize_mesh(minmax_normalized, "2. Min-Max Normalized
[0, 1] range", ax2)
    print("  - Plotting Min-Max Quantized...")
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    visualize_mesh(minmax_quantized_float, 
                   f"3. Min-Max Quantized
({N_BINS} bins)", ax3)
    print("  - Plotting Unit Sphere Normalized...")
    ax4 = fig.add_subplot(2, 3, 5, projection='3d')
    visualize_mesh(unitsphere_normalized, "4. Unit Sphere Normalized
[-1, 1] range", ax4, 'plasma')
    print("  - Plotting Unit Sphere Quantized...")
    ax5 = fig.add_subplot(2, 3, 6, projection='3d')
    visualize_mesh(unitsphere_quantized_float, 
                   f"5. Unit Sphere Quantized
({N_BINS} bins)", ax5, 'plasma')
    plt.tight_layout()
    visualization_path = os.path.join(OUTPUT_DIR, "00_normalization_comparison.png")
    plt.savefig(visualization_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {visualization_path}")
    plt.show()
    
    print("
" + "="*80)
    print("[STEP 7] NORMALIZATION METHOD COMPARISON")
    print("="*80)
    
    comparison_report = """
MIN-MAX NORMALIZATION:


UNIT SPHERE NORMALIZATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Advantages:
  ✓ Centers mesh at origin (removes translation)
  ✓ Fits within unit sphere (uniform in all directions)
  ✓ Preserves geometric relationships and proportions uniformly
  ✓ More robust to different mesh scales
  ✓ Better for comparing multiple meshes with different sizes
  ✓ Provides invariance to translation

Disadvantages:
  ✗ Slightly more complex computation (centroid + distance norm)
  ✗ Range is [-1, 1], requiring shift to [0, 1] for quantization
  ✗ Sensitive to outliers that affect max_distance calculation
  ✗ May lose information if mesh is highly anisotropic

Use Case:
  Best for: Scale-independent normalization, mesh comparison tasks,
  geometrically-consistent transformations, and when rotation invariance 
  is important.


QUANTIZATION ANALYSIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
With 1024 bins (10 bits per axis):
  • Theoretical precision: 1/1023 ≈ 0.00098 per axis
  • Total combinations: 1024³ ≈ 1.07 billion possible states
  • Memory reduction: 32-bit float → 10 bits per coordinate

For increased accuracy, use more bins:
  • 2048 bins (11 bits):  1/2047 ≈ 0.000488 precision
  • 4096 bins (12 bits):  1/4095 ≈ 0.000244 precision
  • 16384 bins (14 bits): 1/16383 ≈ 0.000061 precision


RECOMMENDATION FOR THIS ASSIGNMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Both methods are valid and produce good results. The choice depends on:

1. If your mesh has uniform scaling → Use MIN-MAX (simpler)
2. If comparing multiple different-sized meshes → Use UNIT SPHERE (more consistent)
3. If geometric consistency is critical → Use UNIT SPHERE
4. If simplicity and speed matter → Use MIN-MAX

For the SeamGPT context (3D mesh processing for AI):
  → UNIT SPHERE is recommended because it provides consistent 
     normalization across diverse meshes of different scales and 
     orientations, which is crucial for AI model training stability.
    """
    
    print(comparison_report)
    
    # Save comparison report
    report_path = os.path.join(OUTPUT_DIR, "task2_comparison_report.txt")
    with open(report_path, 'w') as f:
        f.write(comparison_report)
    print(f"
✓ Saved comparison report to: {report_path}")
    
    # ============= STEP 8: SUMMARY =============
    print("
" + "="*80)
    print("TASK 2 SUMMARY")
    print("="*80)
    
    summary = f"""
OUTPUT FILES GENERATED:
─────────────────────────────────────────────────────────────────────────────
1. 01_minmax_normalized.ply          → Min-Max normalized mesh
2. 02_minmax_quantized.ply           → Min-Max normalized + quantized mesh
3. 03_unitsphere_normalized.ply      → Unit Sphere normalized mesh
4. 04_unitsphere_quantized.ply       → Unit Sphere normalized + quantized mesh
5. 00_normalization_comparison.png   → Visual comparison of all methods
6. task2_comparison_report.txt       → Detailed comparison analysis

STATISTICS:
─────────────────────────────────────────────────────────────────────────────
Original vertices:        {original_vertices.shape[0]} points
Quantization bins:        {N_BINS} (10 bits per axis)
Output directory:         {OUTPUT_DIR}

METHODS APPLIED:
─────────────────────────────────────────────────────────────────────────────
✓ Min-Max Normalization   : Scales to [0, 1] per axis independently
✓ Unit Sphere Normalization: Centers and scales to unit sphere [-1, 1]
✓ Quantization             : Discretizes to {N_BINS} bins with 10-bit depth
✓ Visualization            : 3D scatter plots with color mapping
✓ Mesh Saving              : PLY format for all stages

NEXT STEPS:
─────────────────────────────────────────────────────────────────────────────
Proceed to Task 3 to:
  1. Dequantize the quantized meshes
  2. Denormalize back to original coordinate system
  3. Measure reconstruction errors (MSE, MAE)
  4. Analyze per-vertex and per-axis error distributions
  5. Visualize error patterns and compare methods
    """
    
    print(summary)
    summary_path = os.path.join(OUTPUT_DIR, "task2_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"
✓ Saved summary to: {summary_path}")
    
    print("
" + "="*80)
    print("✓ TASK 2 COMPLETED SUCCESSFULLY")
    print("="*80)

if __name__ == "__main__":
    main()
