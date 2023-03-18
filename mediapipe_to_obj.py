# Modified from main.ipynb at https://github.com/apple2373/mediapipe-facemesh

import os
import json
import cv2
import numpy as np
import mediapipe as mp
import skimage
from skimage.transform import PiecewiseAffineTransform, warp
import argparse

# borrowed from https://github.com/YadiraF/DECA/blob/f84855abf9f6956fb79f3588258621b363fa282c/decalib/utils/util.py
def load_obj(obj_filename):
    """ Ref: https://github.com/facebookresearch/pytorch3d/blob/25c065e9dafa90163e7cec873dbb324a637c68b7/pytorch3d/io/obj_io.py
    Load a mesh from a file-like object.
    """
    with open(obj_filename, 'r') as f:
        lines = [line.strip() for line in f]

    verts, uvcoords = [], []
    faces, uv_faces = [], []
    # startswith expects each line to be a string. If the file is read in as
    # bytes then first decode to strings.
    if lines and isinstance(lines[0], bytes):
        lines = [el.decode("utf-8") for el in lines]

    for line in lines:
        tokens = line.strip().split()
        if line.startswith("v "):  # Line is a vertex.
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)
        elif line.startswith("vt "):  # Line is a texture.
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError(
                    "Texture %s does not have 2 values. Line: %s" % (str(tx), str(line))
                )
            uvcoords.append(tx)
        elif line.startswith("f "):  # Line is a face.
            # Update face properties info.
            face = tokens[1:]
            face_list = [f.split("/") for f in face]
            for vert_props in face_list:
                # Vertex index.
                faces.append(int(vert_props[0]))
                if len(vert_props) > 1:
                    if vert_props[1] != "":
                        # Texture index is present e.g. f 4/1/1.
                        uv_faces.append(int(vert_props[1]))

    verts = np.array(verts)
    uvcoords = np.array(uvcoords)
    faces = np.array(faces); faces = faces.reshape(-1, 3) - 1
    uv_faces = np.array(uv_faces); uv_faces = uv_faces.reshape(-1, 3) - 1
    return (
        verts,
        uvcoords,
        faces,
        uv_faces
    )

# borrowed from https://github.com/YadiraF/DECA/blob/f84855abf9f6956fb79f3588258621b363fa282c/decalib/utils/util.py
def write_obj(obj_name,
              vertices,
              faces,
              texture_name = "texture.jpg",
              colors=None,
              texture=None,
              uvcoords=None,
              uvfaces=None,
              inverse_face_order=False,
              normal_map=None,
              ):
    ''' Save 3D face model with texture. 
    Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        faces: shape = (ntri, 3)
        texture: shape = (uv_size, uv_size, 3)
        uvcoords: shape = (nver, 2) max value<=1
    '''
    if os.path.splitext(obj_name)[-1] != '.obj':
        obj_name = obj_name + '.obj'
    mtl_name = obj_name.replace('.obj', '.mtl')
    texture_name
    material_name = 'FaceTexture'

    faces = faces.copy()
    # mesh lab start with 1, python/c++ start from 0
    faces += 1
    if inverse_face_order:
        faces = faces[:, [2, 1, 0]]
        if uvfaces is not None:
            uvfaces = uvfaces[:, [2, 1, 0]]

    # write obj
    with open(obj_name, 'w') as f:
        # first line: write mtlib(material library)
        # f.write('# %s\n' % os.path.basename(obj_name))
        # f.write('#\n')
        # f.write('\n')
        if texture is not None:
            f.write('mtllib %s\n\n' % os.path.basename(mtl_name))

        # write vertices
        if colors is None:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        else:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))

        # write uv coords
        if texture is None:
            for i in range(faces.shape[0]):
                f.write('f {} {} {}\n'.format(faces[i, 2], faces[i, 1], faces[i, 0]))
        else:
            for i in range(uvcoords.shape[0]):
                f.write('vt {} {}\n'.format(uvcoords[i,0], uvcoords[i,1]))
            f.write('usemtl %s\n' % material_name)
            # write f: ver ind/ uv ind
            uvfaces = uvfaces + 1
            for i in range(faces.shape[0]):
                f.write('f {}/{} {}/{} {}/{}\n'.format(
                    #  faces[i, 2], uvfaces[i, 2],
                    #  faces[i, 1], uvfaces[i, 1],
                    #  faces[i, 0], uvfaces[i, 0]
                    faces[i, 0], uvfaces[i, 0],
                    faces[i, 1], uvfaces[i, 1],
                    faces[i, 2], uvfaces[i, 2]
                )
                )
            # write mtl
            with open(mtl_name, 'w') as f:
                f.write('newmtl %s\n' % material_name)
                s = 'map_Kd {}\n'.format(os.path.basename(texture_name)) # map to image
                f.write(s)

                if normal_map is not None:
                    name, _ = os.path.splitext(obj_name)
                    normal_name = f'{name}_normals.png'
                    f.write(f'disp {normal_name}')
                    # out_normal_map = normal_map / (np.linalg.norm(
                    #     normal_map, axis=-1, keepdims=True) + 1e-9)
                    # out_normal_map = (out_normal_map + 1) * 0.5

                    cv2.imwrite(
                        normal_name,
                        # (out_normal_map * 255).astype(np.uint8)[:, :, ::-1]
                        normal_map
                    )
            skimage.io.imsave(texture_name, texture)

def normalize_keypoints(keypoints3d):
    # Rotates and centers the points.
    # One of the rotations is a little too far
    #  Landmarks labeled
    #  https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    center = keypoints3d[0]
    keypoints3d = keypoints3d - center
    axis1 = keypoints3d[165] - keypoints3d[391] # 165 = left side between nose and lip, # right side between nose and lip (one vertex off center)
    axis2 = keypoints3d[2] - keypoints3d[0] # 2 = below nose, 0 = above lip (only 1 vertex apart)
    axis3 = np.cross(axis2,axis1)
    axis3 = axis3/np.linalg.norm(axis3)
    axis2 = axis2/np.linalg.norm(axis2)
    axis1 = np.cross(axis3, axis2)
    axis1 = axis1/np.linalg.norm(axis1)
    U = np.array([axis3,axis2,axis1])
    keypoints3d = keypoints3d.dot(U)
    keypoints3d = keypoints3d - keypoints3d.mean(axis=0) # Doesn't seem to make a difference
    return keypoints3d

def main():
    parser = argparse.ArgumentParser(prog="Mediapipe to OBJ", description="Covert 2D pictures to 3D meshes")
    parser.add_argument('-i', '--input', required=False, help="The path for the face image")
    parser.add_argument('-o', '--output', required=False, help="The output directory. Defaults to 'results/<name of image>.obj'")
    args = parser.parse_args()

    img_path = ''
    if not 'input' in args or args.input == None:
        img_path = input("Filename: ")
    else:
        img_path = args.input
    img_ori = skimage.io.imread(img_path)
    uv_path = "data/uv_map.json" # taken from https://github.com/spite/FaceMeshFaceGeometry/blob/353ee557bec1c8b55a5e46daf785b57df819812c/js/geometry.js
    uv_map_dict = json.load(open(uv_path))
    uv_map = np.array([ (uv_map_dict["u"][str(i)],uv_map_dict["v"][str(i)]) for i in range(468)])

    img = img_ori
    H,W,_ = img.shape
    # run facial landmark detection
    with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(img)

    assert len(results.multi_face_landmarks)==1 

    face_landmarks = results.multi_face_landmarks[0]
    keypoints = np.array([(W*point.x,H*point.y) for point in face_landmarks.landmark[0:468]])#after 468 is iris or something else

    # TODO: Debugging - Save a copy of the image with the points over it

    H_new,W_new = 512,512
    keypoints_uv = np.array([(W_new*x, H_new*y) for x,y in uv_map])

    tform = PiecewiseAffineTransform()
    tform.estimate(keypoints_uv,keypoints)
    texture = warp(img_ori, tform, output_shape=(H_new,W_new))
    texture = (255*texture).astype(np.uint8)
    
    # The X, Y, and Z coords are normalized to 0.0 to 1.0 for the width and height of the image (Z is at the same scale as X).
    # To restore the face to it's original ratio, the X and Z coordinates need to be scaled by the ratio of width to height
    # See https://google.github.io/mediapipe/solutions/face_mesh#output for more details

    width_ratio = W / H
    keypoints3d = np.array([(width_ratio * point.x, point.y, width_ratio * point.z) for point in face_landmarks.landmark[0:468]]) 

    obj_filename = "./data/canonical_face_model.obj"
    canonical_verts,uvcoords,faces,uv_faces = load_obj(obj_filename)

    # keypoints3d already has a face that's more round than the original
    vertices = normalize_keypoints(keypoints3d)
    # vertices = keypoints3d
    
    filename =  os.path.splitext(os.path.basename(img_path))[0] # the name without the extension
    obj_name =  "./results/%s.obj" % filename
    texture_name = "./results/%s_texture.jpg" % filename
    if 'output' in args and args.output != None:
        output_filename = os.path.splitext(args.output)[0]
        obj_name = '%s.obj' % output_filename
        texture_name = '%s_texture.jpg' % output_filename

    save_dir = os.path.split(obj_name)[0]
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # borrowed from https://github.com/YadiraF/PRNet/blob/master/utils/write.py
    write_obj(obj_name,
                vertices,
                faces,
                texture_name,
                texture=texture,
                uvcoords=uvcoords,
                uvfaces=uv_faces,
                )
    
    print('Process Complete!')

if __name__ == '__main__':
    main()