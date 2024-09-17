import sys
sys.path.append('.')
sys.path.append('..')

import argparse
import io
import numpy as np
import os
import smplx
import trimesh

from tools.cfg_parser import Config
from tools.meshviewer import Mesh
from tools.objectmodel import ObjectModel
from tools.utils import makepath, makelogger
from tools.utils import params2torch
from tools.utils import parse_npz
from tools.utils import to_cpu

def export_sequence(cfg, logger=None):

    grab_path = cfg.grab_path
    out_path = cfg.out_path
    makepath(out_path)

    if logger is None:
        logger = makelogger(log_dir=os.path.join(out_path, 'grab_preprocessing.log'), mode='a').info
    else:
        logger = logger
    logger('Starting to get vertices for GRAB!')

    motion_path = cfg.motion_path

    if out_path is None:
        out_path = grab_path

    outfnamehandmesh = makepath(motion_path.replace(grab_path,out_path).replace('.npz', '_full_export_handmesh.obj'), isfile=True)
    outfnameobjectmesh = makepath(motion_path.replace(grab_path,out_path).replace('.npz', '_full_export_objectmesh.obj'), isfile=True)
    outfnamemotion = makepath(motion_path.replace(grab_path,out_path).replace('.npz', '_right_full_export_motion.npz'), isfile=True)

    seq_data = parse_npz(motion_path)
    n_comps = seq_data['n_comps']
    gender = seq_data['gender']

    T = seq_data.n_frames

    rh_mesh_fn = os.path.join(grab_path, '..', seq_data.rhand.vtemp)
    rh_mesh = Mesh(filename=rh_mesh_fn)

    rh_f = rh_mesh.faces
    rh_vtemp = rh_mesh.vertices

    # Extract hand joint rest position

    rest_hand_global_orient = np.zeros((1, seq_data.rhand.params.global_orient.shape[1]))
    rest_hand_transl = np.zeros((1, seq_data.rhand.params.transl.shape[1]))
    rest_hand_pose = np.zeros((1, seq_data.rhand.params.hand_pose.shape[1]))
    rest_hand_fullpose = np.zeros((1, seq_data.rhand.params.fullpose.shape[1]))

    rest_params = {
        'global_orient': rest_hand_global_orient,
        'transl': rest_hand_transl,
        'hand_pose': rest_hand_pose,
        'fullpose': rest_hand_fullpose
    }

    rh_m = smplx.create(model_path=cfg.model_path,
                    model_type='mano',
                    is_rhand = True,
                    v_template = rh_vtemp,
                    num_pca_comps=n_comps,
                    flat_hand_mean=True,
                    batch_size=T)

    rh_parms = params2torch(seq_data.rhand.params)
    rh_output = rh_m(**rh_parms)
    verts_rh = to_cpu(rh_output.vertices)

    rh_root_global_orientations = seq_data.rhand.params.global_orient
    rh_root_translations = seq_data.rhand.params.transl
    joints_rh_orientations = seq_data.rhand.params.fullpose

    rh_m_rest = smplx.create(model_path=cfg.model_path,
                model_type='mano',
                is_rhand = True,
                v_template = rh_vtemp,
                num_pca_comps=n_comps,
                flat_hand_mean=True,
                batch_size=1)

    rh_rest_params = params2torch(rest_params)
    rh_rest_output = rh_m_rest(**rh_rest_params)
    joints_rh_rest_positions = to_cpu(rh_rest_output.joints)
    joints_rh_rest_positions = joints_rh_rest_positions[0]

    # Recorded translations are relative....in the opposite direction?
    rh_root_translations += joints_rh_rest_positions[0]

    joints_hierarchy = np.array([255, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14], dtype=np.uint8)
    num_joints = len(joints_hierarchy)

    rh_relative_rest_configuration = np.zeros(joints_rh_rest_positions.shape, dtype=joints_rh_rest_positions.dtype)
    for ji in range(num_joints):
        parent_index = joints_hierarchy[ji]

        if parent_index == 255:
            rh_relative_rest_configuration[ji] = joints_rh_rest_positions[ji]
        else:
            rh_relative_rest_configuration[ji] = joints_rh_rest_positions[ji] - joints_rh_rest_positions[parent_index]

    obj_mesh_fn = os.path.join(grab_path, '..', seq_data.object.object_mesh)
    obj_mesh = Mesh(filename=obj_mesh_fn)

    obj_f = obj_mesh.faces
    obj_vtemp = obj_mesh.vertices

    obj_m = ObjectModel(v_template=obj_vtemp,
                        batch_size=T)

    obj_parms = params2torch(seq_data.object.params)
    verts_obj = to_cpu(obj_m(**obj_parms).vertices)
    rot_obj = to_cpu(obj_m(**obj_parms).global_orient)
    trans_obj = to_cpu(obj_m(**obj_parms).transl)
    contacts_obj = seq_data['contact']['object']

    table_mesh_fn = os.path.join(grab_path, '..', seq_data.table.table_mesh)
    table_mesh = Mesh(filename=table_mesh_fn)

    table_f = table_mesh.faces
    table_vtemp = table_mesh.vertices

    table_m = ObjectModel(v_template=obj_vtemp,
                    batch_size=T)

    table_parms = params2torch(seq_data.table.params)
    rot_table = to_cpu(table_m(**table_parms).global_orient)
    trans_table = to_cpu(table_m(**table_parms).transl)

    num_hand_faces = rh_f.shape[0]
    num_hand_vertices = verts_rh.shape[1]

    num_object_faces = obj_f.shape[0]
    num_object_vertices = verts_obj.shape[1]
    verts_obj = verts_obj[0] # use tranlation and rotation for the rest to save space

    hand_trimesh = trimesh.Trimesh(vertices=rh_vtemp, faces=rh_f)
    hand_trimesh.export(outfnamehandmesh)

    obj_trimesh = trimesh.Trimesh(vertices=obj_vtemp, faces=obj_f)
    obj_trimesh.export(outfnameobjectmesh)

    # Use sparse representation for contacts
    contact_frames = np.array([], dtype=np.uint16)

    object_contact_frame_counts = np.array([], dtype=np.uint16)
    object_contact_locations = np.array([], dtype=np.uint32)
    object_contact_values = np.array([], dtype=np.uint8)

    is_valid = True

    for frame in range(T):
        contacts_obj_frame = contacts_obj[frame]
        no_obj_contact = np.all((contacts_obj_frame == 0))

        # if not no_obj_contact and not no_hand_contact:
        if not no_obj_contact:
            contact_frames = np.append(contact_frames, np.array([frame], dtype=np.uint16))

            object_contact_frame_vertex_locations =  np.array(np.where(contacts_obj_frame != 0)[0], dtype=np.uint32)
            object_contact_frame_vertex_values = np.array(contacts_obj_frame[object_contact_frame_vertex_locations],  dtype=np.uint8)

            object_contact_frame_counts = np.append(object_contact_frame_counts, np.array([len(object_contact_frame_vertex_locations)], dtype=np.uint16))
            object_contact_locations = np.append(object_contact_locations, object_contact_frame_vertex_locations)
            object_contact_values = np.append(object_contact_values, object_contact_frame_vertex_values)

    if is_valid:
        dumpMotion = {
            'numFrames': T,
            'handVertices': verts_rh.flatten(),
            'handJointHierarchy': joints_hierarchy.flatten(),
            'handRootOrientations': rh_root_global_orientations.flatten(),
            'handRootTranslations': rh_root_translations.flatten(),
            'handJointOrientations': joints_rh_orientations.flatten(),
            'handJointsRestConfiguration': rh_relative_rest_configuration.flatten(),
            'objectRotations': rot_obj.flatten(),
            'objectTranslations': trans_obj.flatten(),
            'tableRotations': rot_table.flatten(),
            'tableTranslations': trans_table.flatten(),
            'contactFrames': contact_frames.flatten(),
            'objectContactFrameCounts': object_contact_frame_counts.flatten(),
            'objectContactLocations': object_contact_locations.flatten(),
            'objectContactValues': object_contact_values.flatten()
        }

        np.savez(outfnamemotion, **dumpMotion)
        print("Export Completed Successfully")
    else:
        print("Sequence is invalid - failed to export")


if __name__ == '__main__':

    msg = '''
        This code will process the desired motion sequence and output a human readable JSON file.

        Please do the following steps before starting the GRAB dataset processing:
        1. Download GRAB dataset from the website https://grab.is.tue.mpg.de/
        2. Set the grab_path, out_path to the correct folder
        3. Change the configuration file for your desired vertices
        4. In case you need body or hand vertices make sure to set the model_path
            to the models downloaded from smplx website
            '''

    parser = argparse.ArgumentParser(description='GRAB-vertices')

    parser.add_argument('--grab-path', required=True, type=str,
                        help='The path to the downloaded grab data')
    parser.add_argument('--out-path', default=None, type=str,
                        help='The path to the folder to save the dump')
    parser.add_argument('--model-path', required=True, type=str,
                        help='The path to the folder containing the MANO model')
    parser.add_argument('--motion-path', required=True, type=str,
                        help='The path to the NPZ file containing the sequence to dump')

    args = parser.parse_args()

    grab_path = args.grab_path
    out_path = args.out_path
    model_path = args.model_path
    motion_path = args.motion_path

    if out_path is None:
        out_path = grab_path

    cfg = {
        # number of vertices samples for each object
        'n_verts_sample': 1024,

        #IO path
        'grab_path': grab_path,
        'out_path': out_path,

        # hand model path
        'model_path':model_path,

        # sequence path
        'motion_path':motion_path
    }

    log_dir = os.path.join(out_path, 'grab_processing.log')
    logger = makelogger(log_dir=log_dir, mode='a').info
    logger(msg)

    cfg = Config(**cfg)

    export_sequence(cfg, logger)
