import numpy as np
import sys
import json
import os
import time
import random
import itertools
from collections import namedtuple, OrderedDict

import bpy
from mathutils import Vector
import bpy_extras
import argparse
import numpy

import os,sys
import os.path as osp
# sys.path.insert(0, '/usr/local/lib/python3.7/site-packages')
# import matplotlib.pyplot as plt


# Don't forget:
# - Set the renderer to Cycles
# - A ground plane set as shadow catcher
# - The compositing nodes should be [Image, RenderLayers] -> AlphaOver -> Composite 
# - The world shader nodes should be Sky Texture -> Background -> World Output
# - Set a background image node

if ".blend" in os.path.realpath(__file__):
    # If the .py has been packaged in the .blend
    curdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
else:
    curdir = os.path.dirname(os.path.realpath(__file__))

class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:] # the list after '--'
        except ValueError as e: # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())



def setScene():
    bpy.data.scenes["Scene"].cycles.film_transparent = True
    try:
        # 2.78 and previous
        #bpy.context.user_preferences.system.compute_device_type = 'CUDA'
        #bpy.context.user_preferences.system.compute_device = 'CUDA_0'
        
        # since 2.79
        prefs = bpy.context.user_preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.compute_device = 'CUDA_0'

        bpy.context.scene.cycles.device = 'GPU'
        print('GPU render!')
    except:
        print('CPU render!')


def getVertices(obj, world=False, first=False):
    """Get the vertices of the object."""
    if first:
        print('-----getVertices')
    vertices = []
    if obj.data:
        if world:
            vertices.append([obj.matrix_world * x.co for x in obj.data.vertices])
        else:
            vertices.append([x.co for x in obj.data.vertices])
        # print('+++++', vertices[-1])
    for idx_child, child in enumerate(obj.children):
        vertices.extend(getVertices(child, world=world))
        print(idx_child)
    return vertices


def getObjBoundaries(obj):
    """Get the object boundary in image space."""
    cam = bpy.data.objects['Camera']
    scene = bpy.context.scene
    list_co = []
    vertices = getVertices(obj, world=True, first=True)
    # vertices_list = []
    # for coord_3d in itertools.chain(*vertices):
    #     vertices_list.append([coord_3d[0], coord_3d[1], coord_3d[2]])
    # vertices_array = np.asarray(vertices_list)
    # vertices_array_center = (np.amin(vertices_list, 0) + np.amax(vertices_list, 0)) / 2.
    # # print('++1', vertices_array.shape, np.amin(vertices_list, 0), np.amax(vertices_list, 0))
    # print('++1', vertices_array_center, obj.location)
    #
    # obj.location = obj.location - Vector((vertices_array_center[0], vertices_array_center[1], vertices_array_center[2]))
    # vertices = getVertices(obj, world=True, first=True)
    # vertices_list = []
    # for coord_3d in itertools.chain(*vertices):
    #     vertices_list.append([coord_3d[0], coord_3d[1], coord_3d[2]])
    # vertices_array = np.asarray(vertices_list)
    # vertices_array_center = (np.amin(vertices_list, 0) + np.amax(vertices_list, 0)) / 2.
    # print('++2', vertices_array.shape, np.amin(vertices_list, 0), np.amax(vertices_list, 0))
    # print('++2', vertices_array_center, obj.location)





    for coord_3d in itertools.chain(*vertices):
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, coord_3d)
        list_co.append([x for x in co_2d])
    list_co = np.asarray(list_co)[:,:2]
    retval = list_co.min(axis=0).tolist()
    retval.extend(list_co.max(axis=0).tolist())
    return retval


def changeVisibility(obj, hide):
    """Hide or show object in render."""
    obj.hide_render = hide
    for child in obj.children:
        changeVisibility(child, hide)


def setCamera(pitch, roll, hfov, vfov, imh, imw, cam_pos=(0, 0, 1.6)):
    # Set camera parameters
    # uses a 35mm camera sensor model
    print('=====setting camera: pitch, roll, hfov, vfov, imh, imw, cam_pos=(0, 0, 1.6)', pitch, roll, hfov, vfov, imh, imw, cam_pos)
    bpy.data.cameras["Camera"].sensor_width = 36
    cam = bpy.data.objects['Camera']
    cam.location = Vector(cam_pos)
    cam.rotation_euler[0] = -pitch + 90.*np.pi/180
    cam.rotation_euler[1] = -roll
    cam.rotation_euler[2] = 0
    # bpy.data.cameras["Camera"].angle_x = hfov
    # bpy.data.cameras["Camera"].angle_y = vfov
    if imh > imw:
        bpy.data.cameras["Camera"].angle = vfov
    else:
        bpy.data.cameras["Camera"].angle = hfov
    bpy.data.scenes["Scene"].render.resolution_x = imw
    bpy.data.scenes["Scene"].render.resolution_y = imh
    bpy.data.scenes["Scene"].render.resolution_percentage = 100
    bpy.data.scenes["Scene"].update()


def setObjectToImagePosition(object_name, ipv, iph):
    """insertion point vertical and horizontal (ipv, iph) in relative units."""

    bpy.data.scenes["Scene"].update()

    cam = bpy.data.objects['Camera']

    # Get the 3D position of the 2D insertion point
    # Get the viewpoint 3D coordinates
    frame = cam.data.view_frame(bpy.context.scene)
    frame = [cam.matrix_world * corner for corner in frame]

    # Perform bilinear interpolation
    top_vec = frame[0] - frame[3]
    bottom_vec = frame[1] - frame[2]
    top_pt = frame[3] + top_vec*iph
    bottom_pt = frame[2] + bottom_vec*iph
    vertical_vec = bottom_pt - top_pt
    unit_location = top_pt + vertical_vec*ipv

    # Find the intersection with the ground plane
    obj_direction = unit_location - cam.location
    length = -cam.location[2]/obj_direction[2]

    # Set the object location
    if len(bpy.data.objects[object_name].children) == 0:
        bpy.data.objects[object_name].location = cam.location + obj_direction*length
        # bpy.data.objects[object_name].location[1] -= 0.25
        # bpy.data.objects[object_name].location[0] -= 0.25
    else:
        for child_obj in bpy.data.objects[object_name].children:
            child_obj.location = cam.location + obj_direction*length

    bpy.data.scenes["Scene"].update()

    print("setObjectToImagePosition: {}".format(bpy.data.objects[object_name].location))


def changeBackgroundImage(bgpath, size):
    if "background" in bpy.data.images:
        previous_background = bpy.data.images["background"]
        bpy.data.images.remove(previous_background)

    img = bpy.data.images.load(bgpath)
    img.name = "background"

    bpy.data.images["background"].scale(*size)

    tree = bpy.context.scene.node_tree
    for node in tree.nodes:
        if isinstance(node, bpy.types.CompositorNodeImage):
            node.image = img
            break
    else:
        raise Exception("Could not find the background image node!")


def setParametricSkyLighting(theta, phi, t):
    """Use the Hosek-Wilkie sky model"""

    # Compute lighting direction
    x = np.sin(theta)*np.sin(phi)
    y = np.sin(theta)*np.cos(phi)
    z = np.cos(theta)

    # Remove previous link to Background and link it with Sky Texture
    link = bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].links[0]
    bpy.data.worlds["World"].node_tree.links.remove(link)
    bpy.data.worlds["World"].node_tree.links.new(
        bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].outputs["Color"],
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0]
    )
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 1.0

    # Set Hosek-Wilkie sky texture
    bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sky_type = "HOSEK_WILKIE"
    bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_direction = Vector((x, y, z))
    bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].turbidity = t
    bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].ground_albedo = 0.3

    bpy.data.objects["Sun"].rotation_euler = Vector((theta, 0, -phi + np.pi))
    bpy.data.lamps["Sun"].shadow_soft_size = 0.03
    bpy.data.lamps["Sun"].node_tree.nodes["Emission"].inputs[1].default_value = 4

    bpy.data.objects["Sun"].hide = False
    bpy.data.objects["Sun"].hide_render = False


def setIBL(path, phi):
    """Use an IBL to light the scene"""

    # Remove previous IBL
    if "envmap" in bpy.data.images:
        previous_background = bpy.data.images["envmap"]
        bpy.data.images.remove(previous_background)

    img = bpy.data.images.load(path)
    img.name = "envmap"

    # Remove previous link to Background and link it with Environment Texture
    link = bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].links[0]
    bpy.data.worlds["World"].node_tree.links.remove(link)
    bpy.data.worlds["World"].node_tree.links.new(
        bpy.data.worlds["World"].node_tree.nodes["Environment Texture"].outputs["Color"],
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0]
    )
    bpy.data.worlds["World"].node_tree.nodes["Environment Texture"].image = img
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 2.0
    bpy.data.worlds["World"].node_tree.nodes["Mapping"].rotation = Vector((0, 0, phi + np.pi/2))

    bpy.data.objects["Sun"].hide = True
    bpy.data.objects["Sun"].hide_render = True


def performRendering(k, suffix="", subfolder="render", close_blender=False, tmp_code=''):

    # Flush scene modifications
    bpy.data.scenes["Scene"].update()

    os.makedirs(subfolder, exist_ok=True)

    # redirect output to log file
    logfile = 'blender_render.log'
    open(logfile, 'a').close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY)

    # do the rendering
    imgpath = os.path.join(curdir, '{}/{}{}_{}.png'.format(subfolder, k, suffix, tmp_code))
    #assert not os.path.isfile(imgpath)
    bpy.data.scenes["Scene"].render.filepath = imgpath
    bpy.ops.render.render(write_still=True)

    # disable output redirection
    os.close(1)
    os.dup(old)
    os.close(old)

    print('>>> Rendered file in ' + imgpath)

    if close_blender:
        bpy.ops.wm.quit_blender()

    # img_render = plt.imread(imgpath)
    # plt.imshow(imgpath)
    # plt.show()


if __name__ == '__main__':
    parser = ArgumentParserForBlender()
    # parser = argparse.ArgumentParser(description="Rui's Scale Estimation Network Training")
    parser.add_argument('-img_path', type=str, default='tmp', help='')
    parser.add_argument('-tmp_code', type=str, default='iamgroot', help='')
    parser.add_argument('-npy_path', type=str, default='iamgroot', help='')
    parser.add_argument('-if_grid', type=bool, default=True, help='if_render grid of cylinders')
    parser.add_argument('-H', type=int, help='')
    parser.add_argument('-W', type=int, help='')
    parser.add_argument('-pitch', type=float, default=0.9, help='')
    parser.add_argument('-fov_v', type=float, default=0.9, help='')
    parser.add_argument('-fov_h', type=float, default=0.9, help='')
    parser.add_argument('-cam_h', type=float, default=0.9, help='')
    parser.add_argument('-insertion_points_x', type=float, default=-1, help='')
    parser.add_argument('-insertion_points_y', type=float, default=-1, help='')
    # parser.add_argument('--pitch', type=float, default=0.9, help='')
    opt = parser.parse_args()

    img_path = opt.img_path
    tmp_code = opt.tmp_code
    imh = opt.H
    imw = opt.W
    insertion_points_x, insertion_points_y = opt.insertion_points_x, opt.insertion_points_y
    pitch = opt.pitch
    roll = 0.
    hfov = opt.fov_h
    vfov = opt.fov_v
    h_cam = opt.cam_h

    setScene()

    insertion_points = [
        (insertion_points_y, insertion_points_x),
        # (400, 360),
        # (320, 320),
        # (430, 200),
    ]
    if insertion_points_x == -1:
        insertion_points_xy_list = np.load(osp.join(opt.npy_path, 'tmp_insert_pts_%s.npy'%tmp_code))
        insertion_points = [(item[1], item[0]) for item in insertion_points_xy_list]

    bbox_hs_list = np.load(osp.join(opt.npy_path, 'tmp_bbox_hs_%s.npy'%tmp_code))

    changeBackgroundImage(img_path, (imw, imh))
    setParametricSkyLighting(np.pi/4, np.pi/8, 3)

    # object_name = "Cone"
    object_name = "Cylinder"
    # object_name = "chair"
    all_obj_names = ['Cone', 'chair', 'Cylinder']


    # pitch, roll, horizontal FoV, image height + width, camera position

    # pitch = -0.11466901
    # roll = 0.
    # hfov = 0.6782765073200586
    # vfov = 0.44870973
    # h_cam = 1.7074147

    # pitch = 1*np.pi/16
    # roll = -np.pi/24
    # hfov = 90*np.pi/180.
    # h_cam = 1.60
    # print('>>>>> Location', object_name, bpy.data.objects[object_name].location)
    # bpy.data.objects[object_name].location = bpy.data.objects['Cone'].location
    # print('>>>>> Location', object_name, bpy.data.objects[object_name].location)
    setCamera(-pitch, roll, hfov, vfov, imh, imw, cam_pos=(0, 0, h_cam))
    changeVisibility(bpy.data.objects[object_name], hide=False)      

    src_obj = bpy.data.objects[object_name]
    for idx, ((ipv, iph), bbox_h) in enumerate(zip(insertion_points, bbox_hs_list)):
        print('===============================', idx)
        # Rotate the object randomly about its y-axis
        # (just for the sake of example, won't do anything on a torus, of course...)
        new_obj = src_obj.copy()
        new_obj.name = '%s_%d'%(object_name, idx)
        new_obj.data = src_obj.data.copy()
        # new_obj.animation_data_clear()
        bpy.context.scene.objects.link(new_obj)
        # obj.rotation_euler[2] = np.random.rand()*2*np.pi
        # new_obj.rotation_euler[2] = np.pi/4.
        # new_obj.location[0] = 10.
        # new_obj.location[1] = 0.
        print('new_obj.location', new_obj.location)

        setObjectToImagePosition(new_obj.name, ipv/imh, iph/imw)

        # Check if object is inside the frame. If not, resize it a tad
        original_scale = new_obj.scale.copy()
        # for tries in range(10):
        #     bpy.data.scenes["Scene"].update()
        #     obj_bounds = getObjBoundaries(new_obj)
        #     print('--', new_obj.name, tries, obj_bounds)
        #     if any(x < 0 for x in obj_bounds[:2]) or any(x > 1 for x in obj_bounds[2:]):
        #         continue
        #         new_obj.scale *= 0.87
        #         print('resized!!!!!!!!!!!!!')
        #     else:
        #         print('Skipping ', idx)
        #         break
        # else:
        #     new_obj.scale = original_scale
        #     bpy.data.scenes["Scene"].update()
        #     print("Object outside boundary! Discarding...")
        #     continue

        bpy.data.scenes["Scene"].update()
        obj_bounds = getObjBoundaries(new_obj)
        print('---obj_bounds', obj_bounds)
        # if any(x < 0 for x in obj_bounds[:2]) or any(x > 1 for x in obj_bounds[2:]):
        #     print('Skipping ', idx)
        #     changeVisibility(new_obj, hide=True)
        #     continue

        if object_name == "Cylinder":
            # print("Object scale: ", new_obj.scale[0] / original_scale[0])
            print('OOOOriginal scale', new_obj.scale)
            print('OOOOriginal dimensions', new_obj.dimensions)
            # # new_obj.scale = new_obj.scale * bbox_h
            new_obj.dimensions = new_obj.dimensions * bbox_h
            # new_obj.dimensions[0] = new_obj.dimensions[0] * 10.
            # new_obj.dimensions[1] *= 10.
            bpy.context.scene.update()
            # # new_obj.scale = Vector((0.1, 0.1, bbox_h))
            print('Afterrrrr scale, dimensions, bbox_h', new_obj.scale, new_obj.dimensions, bbox_h)
            print('Afterrrrr location', new_obj.location)

        # Set on ground, useful if scaled
        vertices = getVertices(new_obj, world=True)
        dist_to_ground = min(v.z for v in itertools.chain(*vertices))
        new_obj.location[2] -= dist_to_ground
        print("Moved the object in Z-axis by", dist_to_ground)
        new_obj.rotation_euler[2] = np.pi/4.
        bpy.context.scene.update()
        print('Afterrrrr location 2', new_obj.location)



        # # Reset scale
        # bpy.data.objects[object_name].scale = original_scale
        # bpy.data.scenes["Scene"].update()
        #

    changeVisibility(bpy.data.objects[object_name], hide=True)
    for obj_name_single in all_obj_names:
        if object_name != obj_name_single:
            changeVisibility(bpy.data.objects[obj_name_single], hide=True)
            print('Hid %s'%obj_name_single)    

    ts = time.time()
    performRendering("render_{}".format('all'), close_blender=len(insertion_points)==1, tmp_code=tmp_code)
    print("Rendering done in {0:0.3f}s".format(time.time() - ts))
    print('Camera location', bpy.data.objects['Camera'].location)
    print("------------------------------")

