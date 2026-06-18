import os
import shutil
import bpy
from . import blender_nerf_operator


# subset of frames operator class
class SubsetOfFrames(blender_nerf_operator.BlenderNeRF_Operator):
    '''Subset of Frames Operator'''
    bl_idname = 'object.subset_of_frames'
    bl_label = 'Subset of Frames SOF'

    def execute(self, context):
        scene = context.scene
        camera = scene.camera

        
        if camera is None:
            self.report({'ERROR'}, 'No camera selected! Please select a camera for SOF method.')
            return {'CANCELLED'}
        
        # Validate camera object
        try:
            if not hasattr(camera, 'data'):
                self.report({'ERROR'}, 'Selected object is not a valid camera!')
                return {'CANCELLED'}
        except AttributeError:
            self.report({'ERROR'}, 'Camera validation failed!')
            return {'CANCELLED'}

        # if there is an error, print first error message
        error_messages = self.asserts(scene, method='SOF')
        if len(error_messages) > 0:
           self.report({'ERROR'}, error_messages[0])
           return {'FINISHED'}

        output_data = self.get_camera_intrinsics(scene, camera)

        # clean directory name (unsupported characters replaced) and output path
        output_dir = bpy.path.clean_name(scene.sof_dataset_name)
        output_path = os.path.join(scene.save_path, output_dir)
        os.makedirs(output_path, exist_ok=True)

        if scene.logs: self.save_log_file(scene, output_path, method='SOF')
        
        # For COLMAP: generate metadata files but let rendering happen normally
        if scene.export_format == 'COLMAP':
            # Generate COLMAP files (cameras, images, points3D)
            self.save_colmap_format(scene, output_path, method='SOF')
        else:
            # Traditional NeRF/NGP export
            if scene.splats: self.save_splats_ply(scene, output_path, method='SOF')

        # Generate JSON transforms for traditional formats
        if scene.export_format != 'COLMAP':
            if scene.test_data:
                # testing transforms
                output_data['frames'] = self.get_camera_extrinsics(scene, camera, mode='TEST', method='SOF')
                self.save_json(output_path, 'transforms_test.json', output_data)

            if scene.train_data:
                # training transforms
                output_data['frames'] = self.get_camera_extrinsics(scene, camera, mode='TRAIN', method='SOF')
                self.save_json(output_path, 'transforms_train.json', output_data)

        # initial properties might have changed since set_init_props update
        scene.init_frame_step = scene.frame_step
        scene.init_output_path = scene.render.filepath

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
            
            scene.rendering = (True, False, False)
            scene.frame_step = scene.train_frame_steps # update frame step
            bpy.ops.render.render('INVOKE_DEFAULT', animation=True, write_still=True) # render scene

        # if frames are rendered, the below code is executed by the handler function
        if not any(scene.rendering):
            # compress dataset and remove folder (only keep zip)
            shutil.make_archive(output_path, 'zip', output_path) # output filename = output_path
            shutil.rmtree(output_path)

        return {'FINISHED'}