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
    rhand_smplx_correspondence_fn = cfg.mano_correspondence
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
    outfnamemotion = makepath(motion_path.replace(grab_path,out_path).replace('.npz', '_full_export_motion.npz'), isfile=True)
    outfnamecontacts = makepath(motion_path.replace(grab_path,out_path).replace('.npz', '_full_export_contacts.json'), isfile=True)

    seq_data = parse_npz(motion_path)
    n_comps = seq_data['n_comps']
    gender = seq_data['gender']

    T = seq_data.n_frames

    rh_mesh_fn = os.path.join(grab_path, '..', seq_data.rhand.vtemp)
    rh_mesh = Mesh(filename=rh_mesh_fn)

    rh_f = rh_mesh.faces
    rh_vtemp = rh_mesh.vertices

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
    joints_rh = to_cpu(rh_output.joints)

    joints_hierarchy = np.array([255, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14], dtype=np.uint8)
    num_joints = len(joints_hierarchy)

    # Convert all joint offsets to relative based on hierarchy
    for frame in range(T):
        joints_frame = joints_rh[frame]
        relative_frame = np.zeros(joints_frame.shape, dtype=joints_frame.dtype)

        for ji in range(num_joints):
            parent_index = joints_hierarchy[ji]

            if parent_index == 255:
                relative_frame[ji] = joints_frame[ji]
            else:
                relative_frame[ji] = joints_frame[ji] - joints_frame[parent_index]

        joints_rh[frame] = relative_frame

    rhand_smplx_correspondence_ids = np.load(rhand_smplx_correspondence_fn)
    contacts_rh = seq_data['contact']['body'][:,rhand_smplx_correspondence_ids]

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

    hand_contact_frame_counts = np.array([], dtype=np.uint16)
    hand_contact_locations = np.array([], dtype=np.uint32)
    hand_contact_values = np.array([], dtype=np.uint8)

    is_valid = True

    for frame in range(T):
        contacts_obj_frame = contacts_obj[frame]
        contacts_hand_frame = contacts_rh[frame]

        no_obj_contact = np.all((contacts_obj_frame == 0))
        no_hand_contact = np.all((contacts_hand_frame == 0))

        if not no_obj_contact and not no_hand_contact:
            contact_frames = np.append(contact_frames, np.array([frame], dtype=np.uint16))

            object_contact_frame_vertex_locations =  np.array(np.where(contacts_obj_frame != 0)[0], dtype=np.uint32)
            object_contact_frame_vertex_values = np.array(contacts_obj_frame[object_contact_frame_vertex_locations],  dtype=np.uint8)

            hand_contact_frame_vertex_locations =  np.array(np.where(contacts_hand_frame != 0)[0], dtype=np.uint32)
            hand_contact_frame_vertex_values = np.array(contacts_hand_frame[hand_contact_frame_vertex_locations],  dtype=np.uint8)

            object_contact_frame_counts = np.append(object_contact_frame_counts, np.array([len(object_contact_frame_vertex_locations)], dtype=np.uint16))
            object_contact_locations = np.append(object_contact_locations, object_contact_frame_vertex_locations)
            object_contact_values = np.append(object_contact_values, object_contact_frame_vertex_values)

            hand_contact_frame_counts = np.append(hand_contact_frame_counts, np.array([len(hand_contact_frame_vertex_locations)], dtype=np.uint16))
            hand_contact_locations = np.append(hand_contact_locations, hand_contact_frame_vertex_locations)
            hand_contact_values = np.append(hand_contact_values, hand_contact_frame_vertex_values)

        elif not no_obj_contact:
            print("Discrepency: Found object contact but no hand contact")
            is_valid = False

        elif not no_hand_contact:
            print("Discrepency: Found hand contact but no object contact")
            is_valid = False

    if is_valid:
        dumpMotion = {
            'numFrames': T,
            'handVertices': verts_rh.flatten(),
            'handJointHierarchy': joints_hierarchy.flatten(),
            'handJoints': joints_rh.flatten(),
            'objectRotations': rot_obj.flatten(),
            'objectTranslations': trans_obj.flatten(),
            'contactFrames': contact_frames.flatten(),
            'objectContactFrameCounts': object_contact_frame_counts.flatten(),
            'objectContactLocations': object_contact_locations.flatten(),
            'objectContactValues': object_contact_values.flatten(),
            'handContactFrameCounts': hand_contact_frame_counts.flatten(),
            'handContactLocations': hand_contact_locations.flatten(),
            'handContactValues': hand_contact_values.flatten()
        }

        np.savez_compressed(outfnamemotion, **dumpMotion)
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

    rhand_smplx_correspondence_fn = os.path.join(grab_path, '..', 'tools', 'smplx_correspondence', 'rhand_smplx_ids.npy')

    cfg = {
        # number of vertices samples for each object
        'n_verts_sample': 1024,

        #IO path
        'grab_path': grab_path,
        'out_path': out_path,

        # hand model path
        'model_path':model_path,

        # sequence path
        'motion_path':motion_path,

        # MANO-SMPLX correspondence
        'mano_correspondence': rhand_smplx_correspondence_fn
    }

    log_dir = os.path.join(out_path, 'grab_processing.log')
    logger = makelogger(log_dir=log_dir, mode='a').info
    logger(msg)

    cfg = Config(**cfg)

    export_sequence(cfg, logger)
