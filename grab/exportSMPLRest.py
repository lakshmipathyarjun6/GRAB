import sys
sys.path.append('.')
sys.path.append('..')

import argparse
import numpy as np
import os
import smplx
import trimesh

import json

from tools.cfg_parser import Config
from tools.meshviewer import Mesh
from tools.utils import makepath, makelogger
from tools.utils import params2torch
from tools.utils import parse_npz
from tools.utils import to_cpu

def export_resting_body(cfg, logger=None):
    
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
        
    outfnamebodymesh = makepath(motion_path.replace(grab_path,out_path).replace('.npz', '_full_export_bodymesh.obj'), isfile=True)
    outfnameskeleton = makepath(motion_path.replace(grab_path,out_path).replace('.npz', '_full_export_skeleton.json'), isfile=True)
    
    seq_data = parse_npz(motion_path)
    n_comps = seq_data['n_comps']
    
    body_mesh_fn = os.path.join(grab_path, '..', seq_data.body.vtemp)
    body_mesh = Mesh(filename=body_mesh_fn)

    body_f = body_mesh.faces
    body_vtemp = body_mesh.vertices
    
    # Extract body joint rest position

    rest_body_global_orient = np.zeros((1, seq_data.body.params.global_orient.shape[1]))
    rest_body_transl = np.zeros((1, seq_data.body.params.transl.shape[1]))
    rest_body_pose = np.zeros((1, seq_data.body.params.body_pose.shape[1]))
    rest_body_fullpose = np.zeros((1, seq_data.body.params.fullpose.shape[1]))

    rest_params = {
        'global_orient': rest_body_global_orient,
        'transl': rest_body_transl,
        'body_pose': rest_body_pose,
        'fullpose': rest_body_fullpose
    }
    
    body_m_rest = smplx.create(model_path=cfg.model_path,
                model_type='smplx',
                v_template = body_vtemp,
                num_pca_comps=n_comps,
                batch_size=1)

    body_rest_params = params2torch(rest_params)
    body_rest_output = body_m_rest(**body_rest_params)
    joints_body_rest_positions = to_cpu(body_rest_output.joints)
    joints_body_rest_positions = joints_body_rest_positions[0]
    
    print(joints_body_rest_positions.shape)
    
    body_trimesh = trimesh.Trimesh(vertices=body_vtemp, faces=body_f)
    body_trimesh.export(outfnamebodymesh)
    
    joints_hierarchy = np.array([255, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19], dtype=np.uint8)
    num_joints = len(joints_hierarchy)

    body_relative_rest_configuration = np.zeros(joints_body_rest_positions.shape, dtype=joints_body_rest_positions.dtype)
    for ji in range(num_joints):
        parent_index = joints_hierarchy[ji]

        if parent_index == 255:
            body_relative_rest_configuration[ji] = joints_body_rest_positions[ji]
        else:
            body_relative_rest_configuration[ji] = joints_body_rest_positions[ji] - joints_body_rest_positions[parent_index]
            
    dumpSkeleton = {
        'handJointHierarchy': joints_hierarchy.flatten().tolist(),
        'bodyJointsRestConfiguration': body_relative_rest_configuration.flatten().tolist()
    }
    
    with open(outfnameskeleton, 'w') as f:
        json.dump(dumpSkeleton, f, indent=4)

    print("Export Completed Successfully")
    
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
                        help='The path to the folder containing the SMPL model')
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

        # body model path
        'model_path':model_path,
        
        # sequence path
        'motion_path':motion_path
    }
    
    log_dir = os.path.join(out_path, 'grab_processing.log')
    logger = makelogger(log_dir=log_dir, mode='a').info
    logger(msg)

    cfg = Config(**cfg)

    export_resting_body(cfg, logger)
