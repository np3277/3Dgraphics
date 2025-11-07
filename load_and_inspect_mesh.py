import trimesh
import numpy as np
def load_and_inspect_mesh(obj_path):
    mesh = trimesh.load(obj_path, process=False)  
    if not isinstance(mesh, trimesh.Trimesh):
        print("Loaded object is not a valid Trimesh object.")
        return
    vertices = mesh.vertices
    faces = mesh.faces
    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    v_mean = vertices.mean(axis=0)
    v_std = vertices.std(axis=0)
    print(f"Mesh file: {obj_path}")
    print(f"Number of vertices: {num_vertices}")
    print(f"Number of faces: {num_faces}")
    print(f"Vertex coordinate statistics:")
    print(f"  Min (x, y, z): {v_min}")
    print(f"  Max (x, y, z): {v_max}")
    print(f"  Mean (x, y, z): {v_mean}")
    print(f"  Std Dev (x, y, z): {v_std}")
    return {
        'vertices': vertices,
        'faces': faces,
        'min': v_min,
        'max': v_max,
        'mean': v_mean,
        'std': v_std
    }
mesh_stats = load_and_inspect_mesh("your_mesh_file.obj")
