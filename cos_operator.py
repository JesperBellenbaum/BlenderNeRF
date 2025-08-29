import os
import shutil
import bpy
from . import helper, blender_nerf_operator


# global addon script variables
EMPTY_NAME = 'BlenderNeRF Sphere'
CAMERA_NAME = 'BlenderNeRF Camera'

# camera on sphere operator class
class CameraOnSphere(blender_nerf_operator.BlenderNeRF_Operator):
    '''Camera on Sphere Operator'''
    bl_idname = 'object.camera_on_sphere'
    bl_label = 'Camera on Sphere COS'

    def execute(self, context):
        scene = context.scene
        camera = scene.camera

        # check if camera is selected : next errors depend on an existing camera
        if camera == None:
            self.report({'ERROR'}, 'Be sure to have a selected camera!')
            return {'FINISHED'}

        # if there is an error, print first error message
        error_messages = self.asserts(scene, method='COS')
        if len(error_messages) > 0:
           self.report({'ERROR'}, error_messages[0])
           return {'FINISHED'}

        output_data = self.get_camera_intrinsics(scene, camera)

        # clean directory name (unsupported characters replaced) and output path
        output_dir = bpy.path.clean_name(scene.cos_dataset_name)
        output_path = os.path.join(scene.save_path, output_dir)
        os.makedirs(output_path, exist_ok=True)

        if scene.logs: self.save_log_file(scene, output_path, method='COS')
        
        # COS camera setup MUST happen for ALL formats
        sphere_camera = None
        if scene.train_data:
            # Set up sphere camera for COS method (required for all formats)
            if not scene.show_camera: 
                scene.show_camera = True
            
            # Robust camera access with error handling - FIX #3
            try:
                if CAMERA_NAME in scene.objects:
                    sphere_camera = scene.objects[CAMERA_NAME]
                    scene.camera = sphere_camera
                else:
                    self.report({'ERROR'}, f'{CAMERA_NAME} not found! COS method requires sphere camera.')
                    return {'CANCELLED'}
            except Exception as e:
                self.report({'ERROR'}, f'Failed to access sphere camera: {e}')
                return {'CANCELLED'}
        
        # Generate format-specific exports
        if scene.export_format == 'COLMAP':
            # Generate COLMAP files (cameras, images, points3D)
            self.save_colmap_format(scene, output_path, method='COS')
        else:
            # Traditional NeRF/NGP export
            if scene.splats: 
                self.save_splats_ply(scene, output_path, method='COS')

        # Generate JSON transforms for traditional formats only
        if scene.export_format != 'COLMAP':
            if scene.test_data:
                # testing transforms (use original camera for test)
                output_data['frames'] = self.get_camera_extrinsics(scene, camera, mode='TEST', method='COS')
                self.save_json(output_path, 'transforms_test.json', output_data)

            if scene.train_data and sphere_camera:
                # training transforms (use sphere camera)
                sphere_output_data = self.get_camera_intrinsics(scene, sphere_camera)
                sphere_output_data['frames'] = self.get_camera_extrinsics(scene, sphere_camera, mode='TRAIN', method='COS')
                self.save_json(output_path, 'transforms_train.json', sphere_output_data)

        # initial properties
        scene.init_output_path = scene.render.filepath
        scene.init_sphere_exists = scene.show_sphere
        scene.init_camera_exists = scene.show_camera
        scene.init_frame_end = scene.frame_end
        scene.init_active_camera = camera

        # Handle rendering (for both COLMAP and traditional formats)
        if scene.train_data and scene.render_frames:
            if scene.export_format == 'COLMAP':
                # For COLMAP: render images to images/ folder (COLMAP standard)
                output_images = os.path.join(output_path, 'images')
                os.makedirs(output_images, exist_ok=True)
                scene.render.filepath = os.path.join(output_images, '')
            else:
                # For traditional: render images to train/ folder
                output_train = os.path.join(output_path, 'train')
                os.makedirs(output_train, exist_ok=True)
                scene.render.filepath = os.path.join(output_train, '')
            
            scene.rendering = (False, False, True)
            scene.frame_end = scene.frame_start + scene.cos_nb_frames - 1 # update end frame
            bpy.ops.render.render('INVOKE_DEFAULT', animation=True, write_still=True) # render scene

        # if frames are rendered, the below code is executed by the handler function
        if not any(scene.rendering):
            # reset camera settings
            if not scene.init_camera_exists: helper.delete_camera(scene, CAMERA_NAME)
            if not scene.init_sphere_exists:
                objects = bpy.data.objects
                objects.remove(objects[EMPTY_NAME], do_unlink=True)
                scene.show_sphere = False
                scene.sphere_exists = False

            scene.camera = scene.init_active_camera

            # compress dataset and remove folder (only keep zip)
            shutil.make_archive(output_path, 'zip', output_path) # output filename = output_path
            shutil.rmtree(output_path)

        return {'FINISHED'}