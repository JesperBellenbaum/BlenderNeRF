import os
import struct
import collections
from mathutils import Matrix
import math


# COLMAP data structures (standalone, no external dependencies)
Camera = collections.namedtuple('Camera', ['id', 'model', 'width', 'height', 'params'])
Image = collections.namedtuple('Image', ['id', 'qvec', 'tvec', 'camera_id', 'name', 'xys', 'point3D_ids'])
Point3D = collections.namedtuple('Point3D', ['id', 'xyz', 'rgb', 'error', 'image_ids', 'point2D_idxs'])

# Camera models supported by COLMAP
CAMERA_MODELS = {
    0: ('SIMPLE_PINHOLE', 3),  # f, cx, cy
    1: ('PINHOLE', 4),         # fx, fy, cx, cy
    2: ('SIMPLE_RADIAL', 4),   # f, cx, cy, k
    3: ('RADIAL', 5),          # f, cx, cy, k1, k2
    4: ('OPENCV', 8),          # fx, fy, cx, cy, k1, k2, p1, p2
}

CAMERA_MODEL_IDS = {v[0]: k for k, v in CAMERA_MODELS.items()}


class ColmapExporter:
    """Standalone COLMAP exporter with no external dependencies"""
    
    @staticmethod
    def write_cameras_text(cameras, path):
        """Write cameras.txt file"""
        with open(path, 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"# Number of cameras: {len(cameras)}\n")
            for camera in cameras:
                params_str = ' '.join(map(str, camera.params))
                f.write(f"{camera.id} {camera.model} {camera.width} {camera.height} {params_str}\n")
    
    @staticmethod
    def write_cameras_binary(cameras, path):
        """Write cameras.bin file"""
        with open(path, 'wb') as f:
            f.write(struct.pack('<Q', len(cameras)))  # Number of cameras
            for camera in cameras:
                model_id = CAMERA_MODEL_IDS[camera.model]
                f.write(struct.pack('<I', camera.id))
                f.write(struct.pack('<I', model_id))
                f.write(struct.pack('<Q', camera.width))
                f.write(struct.pack('<Q', camera.height))
                for param in camera.params:
                    f.write(struct.pack('<d', param))
    
    @staticmethod
    def write_images_text(images, path):
        """Write images.txt file"""
        with open(path, 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {len(images)}, mean observations per image: 0\n")
            for image in images:
                qw, qx, qy, qz = image.qvec
                tx, ty, tz = image.tvec
                f.write(f"{image.id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {image.camera_id} {image.name}\n")
                # Empty POINTS2D line for synthetic data
                f.write("\n")
    
    @staticmethod  
    def write_images_binary(images, path):
        """Write images.bin file"""
        with open(path, 'wb') as f:
            f.write(struct.pack('<Q', len(images)))  # Number of images
            for image in images:
                f.write(struct.pack('<I', image.id))
                
                # Quaternion (w, x, y, z)
                for q in image.qvec:
                    f.write(struct.pack('<d', q))
                
                # Translation (x, y, z)
                for t in image.tvec:
                    f.write(struct.pack('<d', t))
                
                f.write(struct.pack('<I', image.camera_id))
                
                # Image name (null-terminated string)
                f.write(image.name.encode('utf-8') + b'\0')
                
                # Number of 2D points (0 for synthetic data)
                f.write(struct.pack('<Q', 0))
    
    @staticmethod
    def write_points3d_text(points, path):
        """Write points3D.txt file"""
        with open(path, 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write(f"# Number of points: {len(points)}, mean track length: 0\n")
            for point in points:
                x, y, z = point.xyz
                r, g, b = point.rgb
                f.write(f"{point.id} {x} {y} {z} {r} {g} {b} {point.error}\n")
    
    @staticmethod
    def write_points3d_binary(points, path):
        """Write points3D.bin file"""
        with open(path, 'wb') as f:
            f.write(struct.pack('<Q', len(points)))  # Number of points
            for point in points:
                f.write(struct.pack('<Q', point.id))
                
                # XYZ coordinates
                for coord in point.xyz:
                    f.write(struct.pack('<d', coord))
                
                # RGB color
                for color in point.rgb:
                    f.write(struct.pack('<B', color))
                
                # Error
                f.write(struct.pack('<d', point.error))
                
                # Track length (0 for synthetic data)
                f.write(struct.pack('<Q', 0))
    
    @staticmethod
    def write_colmap_model(output_dir, cameras, images, points, binary=True):
        """Write complete COLMAP model (cameras, images, points3D)"""
        os.makedirs(output_dir, exist_ok=True)
        
        if binary:
            ColmapExporter.write_cameras_binary(cameras, os.path.join(output_dir, 'cameras.bin'))
            ColmapExporter.write_images_binary(images, os.path.join(output_dir, 'images.bin'))
            ColmapExporter.write_points3d_binary(points, os.path.join(output_dir, 'points3D.bin'))
        else:
            ColmapExporter.write_cameras_text(cameras, os.path.join(output_dir, 'cameras.txt'))
            ColmapExporter.write_images_text(images, os.path.join(output_dir, 'images.txt'))
            ColmapExporter.write_points3d_text(points, os.path.join(output_dir, 'points3D.txt'))
    
    @staticmethod
    def blender_to_colmap_transform():
        """Rotate 180° around +X: (+X, +Y, -Z) -> (+X, -Y, +Z). Proper rotation (det=+1)."""
        return Matrix.Rotation(math.pi, 4, 'X')
    
    @staticmethod
    def create_camera_from_blender(camera_id, camera_intrinsics, camera_model='SIMPLE_PINHOLE'):
        """Create COLMAP camera from BlenderNeRF camera intrinsics"""
        width = int(camera_intrinsics['w'])
        height = int(camera_intrinsics['h'])
        
        if camera_model == 'SIMPLE_PINHOLE':
            # Use fl_x as focal length, cx, cy as principal point
            params = [
                camera_intrinsics['fl_x'],
                camera_intrinsics['cx'], 
                camera_intrinsics['cy']
            ]
        elif camera_model == 'PINHOLE':
            # Use fl_x, fl_y as focal lengths, cx, cy as principal point
            params = [
                camera_intrinsics['fl_x'],
                camera_intrinsics['fl_y'],
                camera_intrinsics['cx'],
                camera_intrinsics['cy']
            ]
        else:
            raise ValueError(f"Unsupported camera model: {camera_model}")
        
        return Camera(
            id=camera_id,
            model=camera_model,
            width=width,
            height=height,
            params=params
        )
    
    @staticmethod
    def create_image_from_blender(image_id, camera_id, frame_data, transform_matrix):
        """Create COLMAP image (W2C quaternion + t) from Blender frame."""
        # Camera-to-World in Blender coords
        blender_c2w = Matrix(frame_data['transform_matrix'])
        R_bl = blender_c2w.to_3x3()                # C2W rotation (Blender world basis)
        C_bl = blender_c2w.to_translation()        # camera center in Blender world

        # Convert Blender world -> COLMAP world with S (proper rotation)
        S = transform_matrix                        # 180° about X
        S3 = S.to_3x3()
        R_c2w_col = S3 @ R_bl @ S3                  # same as S * R * S^T because S is orthonormal & S^T==S
        C_col     = S3 @ C_bl

        # World-to-Camera
        R_w2c = R_c2w_col.transposed()
        t_col = -(R_w2c @ C_col)

        # Quaternion (qw, qx, qy, qz) from R_w2c
        q = R_w2c.to_quaternion().normalized()

        # Image name
        image_name = os.path.basename(frame_data['file_path'])
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.exr')):
            image_name += '.png'

        return Image(
            id=image_id,
            qvec=[q.w, q.x, q.y, q.z],          # COLMAP order: (qw, qx, qy, qz)
            tvec=[t_col.x, t_col.y, t_col.z],   # COLMAP t = -R*C  (NOT the camera center)
            camera_id=camera_id,
            name=image_name,
            xys=[],
            point3D_ids=[]
        )

    
    @staticmethod
    def create_point3d_from_vertex(point_id, world_pos, color):
        """Create COLMAP point3D from mesh vertex"""
        return Point3D(
            id=point_id,
            xyz=[world_pos.x, world_pos.y, world_pos.z],
            rgb=color,  # [r, g, b] as integers 0-255
            error=0.0,  # No error for synthetic data
            image_ids=[],  # Empty track for synthetic data
            point2D_idxs=[]  # Empty track for synthetic data
        )