import sys
sys.path.append('.')
sys.path.append('..')

import argparse
import os
import json
import smplx
import numpy as np

from tools.cfg_parser import Config
from tools.meshviewer import Mesh
from tools.utils import makelogger
from tools.utils import params2torch
from tools.utils import to_cpu

MAYA_TRANS_SCALE_CORRECTION = 1.0 / 100.0 # cm -> m

def skin_sequence(cfg, logger=None):
    out_path = cfg.out_path
    vtemplate_path = cfg.vtemplate_path
    model_path = cfg.model_path
    
    if logger is None:
        logger = makelogger(log_dir=os.path.join(out_path, 'grab_preprocessing.log'), mode='a').info
    else:
        logger = logger
    logger('Starting skinning...')
    
    motion_path = cfg.motion_path
    outfname = os.path.join(out_path, motion_path.split("/")[-1].split(".")[0] + 'Skinned.hkmexp')
    
    with open(motion_path) as f:
        motion_data = json.load(f)
    
    T = motion_data["totalFrames"]
    data = motion_data["data"]
    
    frame_entries = [entry["frame"] for entry in data]
    complete_pose_data = np.array([entry["handState"] for entry in data])
    
    if complete_pose_data.shape[0] != T:
        print("ERROR: Imported frame totals do not match")
        return
    
    transl_data = complete_pose_data[:, :3]
    transl_data *= MAYA_TRANS_SCALE_CORRECTION
    
    global_orient_data = complete_pose_data[:, 3:6]
    
    fullpose_data = complete_pose_data[:, 6:]
    
    rhand_params = {
        'global_orient': global_orient_data,
        'transl': transl_data,
        'fullpose': fullpose_data
    }
    
    rh_mesh = Mesh(filename=vtemplate_path)
    
    rh_vtemp = rh_mesh.vertices
    
    rh_m = smplx.create(model_path=model_path,
                model_type='mano',
                is_rhand = False,
                v_template = rh_vtemp,
                flat_hand_mean=True,
                batch_size=T)
    
    torch_rh_params = params2torch(rhand_params)
    rh_output = rh_m(**torch_rh_params)
    verts_rh = to_cpu(rh_output.vertices).astype(float)
    
    data_entries = []
    
    for i in range(T):
        frame_number = frame_entries[i]
        vertex_data = list(verts_rh[i].flatten())

        data_entry = {
            "frame": frame_number,
            "handState": vertex_data
        }
        
        data_entries.append(data_entry)
    
    skinDump = {
        'handSuffixName': motion_data["handSuffixName"],
        'totalFrames': T,
        'data': data_entries
    }
    
    with open(outfname, "w") as f:
        json.dump(skinDump, f, indent=4)

if __name__ == '__main__':

    msg = '''
        This code will process a JSON mocap file and output MANO skin vertices.

        Please do the following steps before starting the GRAB dataset processing:
        1. Set the grab_path, out_path to the correct folder
        2. Change the configuration file for your desired vertices
        3. In case you need body or hand vertices make sure to set the model_path
            to the models downloaded from smplx website
            '''

    parser = argparse.ArgumentParser(description='GRAB-vertices')
    
    parser.add_argument('--out-path', default=None, type=str,
                        help='The path to the folder to save the skinned data')
    parser.add_argument('--vtemplate-path', default=None, type=str,
                        help='The path to the MANO mesh template')
    parser.add_argument('--model-path', required=True, type=str,
                        help='The path to the folder containing the MANO model')
    parser.add_argument('--motion-path', required=True, type=str,
                        help='The path to the HKMEXP (JSON) file containing the sequence to skin')
    
    args = parser.parse_args()

    out_path = args.out_path
    vtemplate_path = args.vtemplate_path
    model_path = args.model_path
    motion_path = args.motion_path
    
    cfg = {
        # number of vertices samples for each object
        'n_verts_sample': 1024,

        #IO path
        'out_path': out_path,
        
        #Hand mesh template
        'vtemplate_path': vtemplate_path,

        # hand model path
        'model_path':model_path,

        # sequence path
        'motion_path':motion_path
    }

    log_dir = os.path.join(out_path, 'grab_processing.log')
    logger = makelogger(log_dir=log_dir, mode='a').info
    logger(msg)

    cfg = Config(**cfg)

    skin_sequence(cfg, logger)
