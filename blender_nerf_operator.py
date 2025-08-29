import os
import math
import json
import datetime
import bpy
from mathutils import Matrix


# global addon script variables
OUTPUT_TRAIN = 'train'
OUTPUT_TEST = 'test'
CAMERA_NAME = 'BlenderNeRF Camera'
TMP_VERTEX_COLORS = 'blendernerf_vertex_colors_tmp'


# blender nerf operator parent class
class BlenderNeRF_Operator(bpy.types.Operator):

    # camera intrinsics
    def get_camera_intrinsics(self, scene, camera):
        import math
        cam = camera.data

        # Final render size
        scale = scene.render.resolution_percentage / 100.0
        width  = int(round(scene.render.resolution_x * scale))
        height = int(round(scene.render.resolution_y * scale))

        # Pixel aspect (usually 1.0 / 1.0)
        pax = float(scene.render.pixel_aspect_x)
        pay = float(scene.render.pixel_aspect_y)

        # Blender’s FOVs already incorporate lens, sensor size, and sensor_fit.
        ax = cam.angle_x
        ay = cam.angle_y

        # Focal in "pixel units" — account for non-square pixels
        # Wider pixels (pax>1) mean fewer horizontal pixels per same angle -> divide width by pax.
        fx = 0.5 * (width  / pax) / math.tan(0.5 * ax)
        fy = 0.5 * (height / pay) / math.tan(0.5 * ay)

        # Principal point with lens shift (Blender shift is in NDC-ish units of focal lengths)
        # Positive shift_x moves the image right -> principal point moves left.
        cx = 0.5 * width  - cam.shift_x * fx
        # Positive shift_y moves the image up -> principal point moves down.
        cy = 0.5 * height + cam.shift_y * fy

        intr = {
            'camera_angle_x': ax,
            'camera_angle_y': ay,
            'fl_x': fx,
            'fl_y': fy,
            'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0,
            'cx': cx, 'cy': cy,
            'w': width, 'h': height,
            'aabb_scale': scene.aabb,
        }

        # Keep your minimal NERF payload behavior.
        return {'camera_angle_x': ax} if scene.export_format == 'NERF' else intr


    # camera extrinsics (transform matrices)
    def get_camera_extrinsics(self, scene, camera, mode='TRAIN', method='SOF'):
        assert mode == 'TRAIN' or mode == 'TEST'
        assert method == 'SOF' or method == 'TTC' or method == 'COS'

        if scene.splats and scene.splats_test_dummy and mode == 'TEST':
            return []

        initFrame = scene.frame_current
        step = scene.train_frame_steps if (mode == 'TRAIN' and method == 'SOF') else scene.frame_step
        if (mode == 'TRAIN' and method == 'COS'):
            end = scene.frame_start + scene.cos_nb_frames - 1
        elif (mode == 'TRAIN' and method == 'TTC'):
            end = scene.frame_start + scene.ttc_nb_frames - 1
        else:
            end = scene.frame_end

        camera_extr_dict = []
        for frame in range(scene.frame_start, end + 1, step):
            scene.frame_set(frame)
            filename = os.path.basename( scene.render.frame_path(frame=frame) )
            filedir = OUTPUT_TRAIN * (mode == 'TRAIN') + OUTPUT_TEST * (mode == 'TEST')

            frame_data = {
                'file_path': os.path.join(filedir, os.path.splitext(filename)[0] if scene.splats else filename),
                'transform_matrix': self.listify_matrix(camera.matrix_world)
            }

            camera_extr_dict.append(frame_data)

        scene.frame_set(initFrame) # set back to initial frame

        return camera_extr_dict

    # export vertex colors for each visible mesh or COLMAP format
    def save_splats_ply(self, scene, directory, method='SOF'):
        if scene.export_format == 'COLMAP':
            # Export in COLMAP format instead of PLY
            self.save_colmap_format(scene, directory, method)
            return
            
        # Original PLY export logic
        # create temporary vertex colors
        for obj in scene.objects:
            if obj.type == 'MESH':
                if not obj.data.vertex_colors:
                    obj.data.vertex_colors.new(name=TMP_VERTEX_COLORS)

        if bpy.context.object is None or bpy.context.active_object is None:
            self.report({'INFO'}, 'No object active. Setting first object as active.')
            bpy.context.view_layer.objects.active = bpy.data.objects[0]

        init_mode = bpy.context.object.mode
        bpy.ops.object.mode_set(mode='OBJECT')

        init_active_object = bpy.context.active_object
        init_selected_objects = bpy.context.selected_objects
        bpy.ops.object.select_all(action='DESELECT')

        # select only visible meshes
        for obj in scene.objects:
            if obj.type == 'MESH' and self.is_object_visible(obj):
                obj.select_set(True)

        # save ply file
        bpy.ops.wm.ply_export(filepath=os.path.join(directory, 'points3d.ply'), export_normals=True, export_attributes=False, ascii_format=True)

        # remove temporary vertex colors
        for obj in scene.objects:
            if obj.type == 'MESH' and self.is_object_visible(obj):
                if obj.data.vertex_colors and TMP_VERTEX_COLORS in obj.data.vertex_colors:
                    obj.data.vertex_colors.remove(obj.data.vertex_colors[TMP_VERTEX_COLORS])

        bpy.context.view_layer.objects.active = init_active_object
        bpy.ops.object.select_all(action='DESELECT')

        # reselect previously selected objects
        for obj in init_selected_objects:
            obj.select_set(True)

        bpy.ops.object.mode_set(mode=init_mode)

    def _get_method_camera(self, scene, method, mode):
        """Get the appropriate camera for the given method and mode - FIX #2"""
        try:
            if method == 'COS':
                if mode == 'TRAIN' and scene.train_data:
                    # For COS training, use sphere camera if it exists
                    if CAMERA_NAME in scene.objects:
                        return scene.objects[CAMERA_NAME]
                    else:
                        self.report({'WARNING'}, f'{CAMERA_NAME} not found, using default camera')
                        return scene.camera
                else:
                    # For COS testing, use main camera
                    return scene.camera
                    
            elif method == 'TTC':
                if mode == 'TRAIN':
                    if scene.camera_train_target is None:
                        self.report({'ERROR'}, 'Train camera not selected for TTC method')
                        return None
                    return scene.camera_train_target
                elif mode == 'TEST':
                    if scene.camera_test_target is None:
                        self.report({'ERROR'}, 'Test camera not selected for TTC method')
                        return None
                    return scene.camera_test_target
                else:
                    return scene.camera
                    
            else:  # SOF or default
                if scene.camera is None:
                    self.report({'ERROR'}, 'No camera selected')
                    return None
                return scene.camera
                
        except (AttributeError, KeyError) as e:
            self.report({'ERROR'}, f'Camera access error: {e}')
            return None

    def save_colmap_format(self, scene, directory, method='SOF'):
        """Export COLMAP format using standalone exporter (method-aware, robust intrinsics/poses)."""
        import os
        from .colmap_export import ColmapExporter

        # --- paths ----------------------------------------------------------------
        colmap_dir = os.path.join(directory, 'sparse', '0')
        os.makedirs(colmap_dir, exist_ok=True)

        # --- camera selection ------------------------------------------------------
        camera = self._get_method_camera(scene, method, 'TRAIN')
        if camera is None:
            self.report({'ERROR'}, f'Required camera not found for method {method}')
            return

        # --- intrinsics & extrinsics ----------------------------------------------
        camera_intrinsics = self.get_camera_intrinsics(scene, camera)
        camera_extrinsics = self.get_camera_extrinsics(scene, camera, 'TRAIN', method)
        if not camera_extrinsics:
            self.report({'ERROR'}, 'No frames to export (check frame range / step).')
            return

        # Choose model automatically: PINHOLE if fx != fy
        fx = float(camera_intrinsics['fl_x'])
        fy = float(camera_intrinsics['fl_y'])
        camera_model_name = 'SIMPLE_PINHOLE' if abs(fx - fy) < 1e-9 else 'PINHOLE'

        # Build COLMAP camera
        colmap_camera = ColmapExporter.create_camera_from_blender(
            camera_id=1,
            camera_intrinsics=camera_intrinsics,
            camera_model=camera_model_name
        )
        cameras = [colmap_camera]

        # --- filenames/extensions --------------------------------------------------
        # Blender’s actual render extension (uppercase in UI); enforce PNG for splats.
        ext = scene.render.image_settings.file_format.lower()
        if scene.splats:
            ext = 'png'  # Gaussian Splatting requires PNG in your pipeline

        def ensure_ext(name: str) -> str:
            n = name
            # If the path came without extension (common in your get_camera_extrinsics with splats),
            # add the correct extension. Otherwise, normalize it.
            if not n.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.exr')):
                n = f"{n}.{ext}"
            else:
                # normalize extension to what we actually write on disk
                base, _old = os.path.splitext(n)
                n = f"{base}.{ext}"
            return n

        # --- pose conversion (uses fixed ColmapExporter.create_image_from_blender) --
        S = ColmapExporter.blender_to_colmap_transform()
        images = []
        for i, frame_data in enumerate(camera_extrinsics):
            # Normalize filename to real file on disk
            frame_data = dict(frame_data)  # shallow copy
            frame_data['file_path'] = ensure_ext(frame_data['file_path'])

            img = ColmapExporter.create_image_from_blender(
                image_id=i + 1,
                camera_id=1,
                frame_data=frame_data,
                transform_matrix=S
            )
            images.append(img)

        # --- points from visible meshes -------------------------------------------
        points = []
        point_id = 1
        for obj in scene.objects:
            if obj.type == 'MESH' and self.is_object_visible(obj):
                mesh = obj.data
                matrix_world = obj.matrix_world

                has_vcol = mesh.vertex_colors and len(mesh.vertex_colors) > 0
                vcol_layer = mesh.vertex_colors[0] if has_vcol else None

                for poly in mesh.polygons:
                    for loop_index in poly.loop_indices:
                        v_idx = mesh.loops[loop_index].vertex_index
                        vert = mesh.vertices[v_idx]

                        # Apply the same Blender→COLMAP axis flip as used for cameras
                        world_pos = (ColmapExporter.blender_to_colmap_transform().to_3x3() @ (matrix_world @ vert.co))

                        if has_vcol:
                            cd = vcol_layer.data[loop_index].color
                            color = [int(cd[0] * 255), int(cd[1] * 255), int(cd[2] * 255)]
                        else:
                            color = [128, 128, 128]

                        p = ColmapExporter.create_point3d_from_vertex(
                            point_id=point_id,
                            world_pos=world_pos,
                            color=color
                        )
                        points.append(p)
                        point_id += 1

        # --- write model -----------------------------------------------------------
        binary = scene.colmap_binary
        ColmapExporter.write_colmap_model(colmap_dir, cameras, images, points, binary)

        # --- validations & report --------------------------------------------------
        # 1) Make sure images.txt names match your files. We only check names here;
        #    existence check would require disk access to your render path, which may vary.
        n_imgs = len(images)
        n_pts = len(points)
        cam_model = cameras[0].model
        res = f"{cameras[0].width}x{cameras[0].height}"
        self.report({'INFO'}, f'COLMAP exported to {colmap_dir} | {n_imgs} images, {n_pts} pts | {cam_model} @ {res}')


    def save_json(self, directory, filename, data, indent=4):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=indent)

    def is_power_of_two(self, x):
        return math.log2(x).is_integer()

    # function from original nerf 360_view.py code for blender
    def listify_matrix(self, matrix):
        matrix_list = []
        for row in matrix:
            matrix_list.append(list(row))
        return matrix_list

    # check whether an object is visible in render
    def is_object_visible(self, obj):
        if obj.hide_render:
            return False

        for collection in obj.users_collection:
            if collection.hide_render:
                return False

        return True

    # assert messages
    def asserts(self, scene, method='SOF'):
        assert method == 'SOF' or method == 'TTC' or method == 'COS'

        camera = scene.camera
        train_camera = scene.camera_train_target
        test_camera = scene.camera_test_target

        sof_name = scene.sof_dataset_name
        ttc_name = scene.ttc_dataset_name
        cos_name = scene.cos_dataset_name

        error_messages = []

        if (method == 'SOF' or method == 'COS') and not camera.data.type == 'PERSP':
            error_messages.append('Only perspective cameras are supported!')

        if method == 'TTC' and not (train_camera.data.type == 'PERSP' and test_camera.data.type == 'PERSP'):
           error_messages.append('Only perspective cameras are supported!')

        if method == 'COS' and CAMERA_NAME in scene.objects.keys():
            sphere_camera = scene.objects[CAMERA_NAME]
            if not sphere_camera.data.type == 'PERSP':
                error_messages.append('BlenderNeRF Camera must remain a perspective camera!')

        if (method == 'SOF' and sof_name == '') or (method == 'TTC' and ttc_name == '') or (method == 'COS' and cos_name == ''):
            error_messages.append('Dataset name cannot be empty!')

        if method == 'COS' and any(x == 0 for x in scene.sphere_scale):
            error_messages.append('The BlenderNeRF Sphere cannot be flat! Change its scale to be non zero in all axes.')

        if scene.export_format != 'NERF' and not self.is_power_of_two(scene.aabb):
            error_messages.append('AABB scale needs to be a power of two!')

        if scene.save_path == '':
            error_messages.append('Save path cannot be empty!')

        if scene.splats and not scene.test_data:
            error_messages.append('Gaussian Splatting requires test data!')

        if scene.splats and scene.render.image_settings.file_format != 'PNG':
            error_messages.append('Gaussian Splatting requires PNG file extensions!')

        return error_messages

    def save_log_file(self, scene, directory, method='SOF'):
        assert method == 'SOF' or method == 'TTC' or method == 'COS'
        now = datetime.datetime.now()

        logdata = {
            'BlenderNeRF Version': scene.blendernerf_version,
            'Date and Time' : now.strftime("%d/%m/%Y %H:%M:%S"),
            'Train': scene.train_data,
            'Test': scene.test_data,
            'AABB': scene.aabb,
            'Render Frames': scene.render_frames,
            'File Format': scene.export_format,
            'Save Path': scene.save_path,
            'Method': method
        }

        if method == 'SOF':
            logdata['Frame Step'] = scene.train_frame_steps
            logdata['Camera'] = scene.camera.name
            logdata['Dataset Name'] = scene.sof_dataset_name

        elif method == 'TTC':
            logdata['Train Camera Name'] = scene.camera_train_target.name
            logdata['Test Camera Name'] = scene.camera_test_target.name
            logdata['Frames'] = scene.ttc_nb_frames
            logdata['Dataset Name'] = scene.ttc_dataset_name

        else:
            logdata['Camera'] = scene.camera.name
            logdata['Location'] = str(list(scene.sphere_location))
            logdata['Rotation'] = str(list(scene.sphere_rotation))
            logdata['Scale'] = str(list(scene.sphere_scale))
            logdata['Radius'] = scene.sphere_radius
            logdata['Lens'] = str(scene.focal) + ' mm'
            logdata['Seed'] = scene.seed
            logdata['Frames'] = scene.cos_nb_frames
            logdata['Upper Views'] = scene.upper_views
            logdata['Outwards'] = scene.outwards
            logdata['Dataset Name'] = scene.cos_dataset_name

        self.save_json(directory, filename='log.txt', data=logdata)