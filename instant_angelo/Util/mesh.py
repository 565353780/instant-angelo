"""Mesh extraction utilities"""

import torch
import numpy as np


def extract_mesh_from_sdf(sdf_func, bounds, intv, block_res=64, texture_func=None, filter_lcc=False):
    """从 SDF 函数提取网格
    
    Args:
        sdf_func: SDF 函数 (points -> sdf_values)
        bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        intv: 网格间隔
        block_res: 分块分辨率
        texture_func: 纹理函数（可选）
        filter_lcc: 是否只保留最大连通分量
        
    Returns:
        trimesh.Trimesh: 提取的网格
    """
    import trimesh
    
    try:
        import mcubes
    except ImportError:
        print("Warning: mcubes not installed. Please install with: pip install PyMCubes")
        return trimesh.Trimesh()
    
    bounds = np.array(bounds)
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    z_min, z_max = bounds[2]
    
    # 创建网格
    x = torch.arange(x_min, x_max, intv)
    y = torch.arange(y_min, y_max, intv)
    z = torch.arange(z_min, z_max, intv)
    
    res_x, res_y, res_z = len(x), len(y), len(z)
    print(f"Extracting mesh at resolution {res_x} x {res_y} x {res_z}")
    
    # 分块处理
    num_blocks_x = int(np.ceil(res_x / block_res))
    num_blocks_y = int(np.ceil(res_y / block_res))
    num_blocks_z = int(np.ceil(res_z / block_res))
    
    mesh_blocks = []
    
    for bx in range(num_blocks_x):
        for by in range(num_blocks_y):
            for bz in range(num_blocks_z):
                xi = bx * block_res
                yi = by * block_res
                zi = bz * block_res
                
                x_block = x[xi:xi+block_res+1]
                y_block = y[yi:yi+block_res+1]
                z_block = z[zi:zi+block_res+1]
                
                xx, yy, zz = torch.meshgrid(x_block, y_block, z_block, indexing='ij')
                xyz = torch.stack([xx, yy, zz], dim=-1)
                
                # 计算 SDF
                with torch.no_grad():
                    sdf = sdf_func(xyz.cuda()).cpu()
                    
                # Marching cubes
                sdf_np = sdf.numpy()
                if sdf_np.min() < 0 and sdf_np.max() > 0:
                    vertices, faces = mcubes.marching_cubes(sdf_np, 0)
                    
                    if vertices.shape[0] > 0:
                        # 转换到世界坐标
                        vertices = vertices * intv + np.array([x_block[0], y_block[0], z_block[0]])
                        mesh_block = trimesh.Trimesh(vertices, faces)
                        mesh_blocks.append(mesh_block)
                        
    # 合并所有块
    if mesh_blocks:
        mesh = trimesh.util.concatenate(mesh_blocks)
        
        # 过滤最大连通分量
        if filter_lcc:
            mesh = filter_largest_cc(mesh)
            
        # 添加纹理
        if texture_func is not None and mesh.vertices.shape[0] > 0:
            colors = texture_func(mesh.vertices)
            mesh.visual.vertex_colors = colors
            
        return mesh
    else:
        return trimesh.Trimesh()


def filter_largest_cc(mesh):
    """过滤最大连通分量"""
    import trimesh
    
    if mesh.vertices.shape[0] == 0:
        return mesh
        
    # 获取连通分量
    components = mesh.split(only_watertight=False)
    
    if len(components) == 0:
        return mesh
        
    # 找到最大的连通分量
    largest = max(components, key=lambda x: x.vertices.shape[0])
    return largest


def filter_points_outside_bounding_sphere(mesh, radius=1.0):
    """过滤边界球外的点"""
    import trimesh
    
    if mesh.vertices.shape[0] == 0:
        return mesh
        
    # 计算到原点的距离
    distances = np.linalg.norm(mesh.vertices, axis=1)
    
    # 找到在球内的面
    valid_vertices = distances < radius
    valid_faces = valid_vertices[mesh.faces].all(axis=1)
    
    # 创建新网格
    new_mesh = mesh.submesh([valid_faces], only_watertight=False, append=True)
    
    return new_mesh
