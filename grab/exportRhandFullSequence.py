import sys
sys.path.append('.')
sys.path.append('..')

import argparse
import io
import numpy as np
import os
import smplx

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

    rhand_smplx_correspondence_ids = np.load(rhand_smplx_correspondence_fn)

    if logger is None:
        logger = makelogger(log_dir=os.path.join(out_path, 'grab_preprocessing.log'), mode='a').info
    else:
        logger = logger
    logger('Starting to get vertices for GRAB!')

    motion_path = cfg.motion_path

    if out_path is None:
        out_path = grab_path

    outfname = makepath(motion_path.replace(grab_path,out_path).replace('.npz', '_full_export.npz'), isfile=True)

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
    contacts_rh = seq_data['contact']['body'][:,rhand_smplx_correspondence_ids]

    obj_mesh_fn = os.path.join(grab_path, '..', seq_data.object.object_mesh)
    obj_mesh = Mesh(filename=obj_mesh_fn)

    obj_f = obj_mesh.faces
    obj_vtemp = obj_mesh.vertices

    obj_m = ObjectModel(v_template=obj_vtemp,
                        batch_size=T)

    obj_parms = params2torch(seq_data.object.params)
    verts_obj = to_cpu(obj_m(**obj_parms).vertices)
    contacts_obj = seq_data['contact']['object']

    num_hand_faces = rh_f.shape[0]
    num_hand_vertices = verts_rh.shape[1]

    num_object_faces = obj_f.shape[0]
    num_object_vertices = verts_obj.shape[1]

    dumpData = {
        'numFrames': T,
        'numHandFaces': num_hand_faces,
        'numHandVertices': num_hand_vertices,
        'handFaces': rh_f.flatten(),
        'handVertices': verts_rh.flatten(),
        'handContacts': contacts_rh.flatten(),
        'numObjectFaces': num_object_faces,
        'numObjectVertices': num_object_vertices,
        'objectFaces': obj_f.flatten(),
        'objectVertices': verts_obj.flatten(),
        'objectContacts': contacts_obj.flatten()
    }

    np.savez_compressed(outfname, **dumpData)


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
