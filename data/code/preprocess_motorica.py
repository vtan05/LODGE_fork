import argparse
import os
from pydoc import doc
from cv2 import mean
import numpy as np
from pathlib import Path
import torch
import sys
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append(os.getcwd()) 
from dld.data.render_joints.smplfk import SMPLX_Skeleton, do_smplxfk, ax_to_6v, ax_from_6v


floor_height = 0


def vectorize_many(data):
    # given a list of batch x seqlen x joints? x channels, flatten all to batch x seqlen x -1, concatenate
    batch_size = data[0].shape[0]
    seq_len = data[0].shape[1]

    out = [x.reshape(batch_size, seq_len, -1).contiguous() for x in data]

    global_pose_vec_gt = torch.cat(out, dim=2)
    return global_pose_vec_gt

def set_on_ground(root_pos, local_q_72, smplx_model):
    # root_pos = root_pos[:, :] - root_pos[:1, :]
    length = root_pos.shape[0]
    # model_q = model_q.view(b*s, -1)
    # model_x = model_x.view(-1, 3)
    positions = smplx_model.forward(local_q_72, root_pos)
    positions = positions.view(length, -1, 3)   # bxt, j, 3
    
    l_toe_h = positions[0, 10, 1] - floor_height
    r_toe_h = positions[0, 11, 1] - floor_height
    if abs(l_toe_h - r_toe_h) < 0.02:
        height = (l_toe_h + r_toe_h)/2
    else:
        height = min(l_toe_h, r_toe_h)
    root_pos[:, 1] = root_pos[:, 1] - height

    return root_pos, local_q_72

def set_on_ground_139(data, smplx_model, ground_h=0):
    length = data.shape[0]
    assert len(data.shape) == 2
    assert data.shape[1] == 139
    positions = do_smplxfk(data, smplx_model)
    l_toe_h = positions[0, 10, 1] - floor_height
    r_toe_h = positions[0, 11, 1] - floor_height
    if abs(l_toe_h - r_toe_h) < 0.02:
        height = (l_toe_h + r_toe_h)/2
    else:
        height = min(l_toe_h, r_toe_h)
    data[:, 5] = data[:, 5] - (height -  ground_h)

    return data

def motion_feats_extract(moinputs_dir, mooutputs_dir):

    device = "cpu"
    print("extracting")
    raw_fps = 30
    data_fps = 30
    data_fps <= raw_fps
    device = "cpu"
    smplx_model = SMPLX_Skeleton()

    os.makedirs(mooutputs_dir, exist_ok=True)
        
    motions = sorted(glob.glob(os.path.join(moinputs_dir, "*.npz")))
    for motion in tqdm(motions):
        print(motion)
        data = np.load(motion)
        fname = os.path.basename(motion).split(".")[0]

        pos = np.squeeze(data["trans"]) 
        q = np.squeeze([data["poses"]]) # in axis-angle
        print("pos.shape", pos.shape)
        print("q.shape", q.shape)
        root_pos = torch.Tensor(pos).to(device) 
        local_q = torch.Tensor(q).to(device) 

        length = root_pos.shape[0]
        local_q_72 = local_q.view(length, 72)
        root_pos, local_q_72 = set_on_ground(root_pos, local_q_72, smplx_model)
        positions = smplx_model.forward(local_q_72, root_pos)
        positions = positions.view(length, -1, 3)   

        # contacts

        feet = positions[:, (7, 8, 10, 11)]  
        contacts_d_ankle = (feet[:,:2,1] < 0.12).to(local_q_72)
        contacts_d_teo = (feet[:,2:,1] < 0.05).to(local_q_72)
        contacts_d = torch.cat([contacts_d_ankle, contacts_d_teo], dim=-1).detach().cpu().numpy()


        local_q_72 = local_q_72.view(length, 24, 3)  
        print("local_q_72.shape", local_q_72.shape) # L, 24, 3
        local_q_144 = ax_to_6v(local_q_72).view(length,144).detach().cpu().numpy()
        print("contacts_d.shape", contacts_d.shape) # L, 4
        print("root_pos.shape", root_pos.shape) # L, 3
        print("local_q_144.shape", local_q_144.shape) # L, 144
        mofeats_input = np.concatenate( [contacts_d, root_pos, local_q_144] ,axis=-1)
        np.save(os.path.join(mooutputs_dir, fname+".npy"), mofeats_input)
        print("mofeats_input", mofeats_input.shape) # L, 151
    return


if __name__ == "__main__":

    motion_feats_extract(moinputs_dir=r"/host_data/van/Dance_v2/LODGE/data/motorica/smpl", 
                        mooutputs_dir=r"/host_data/van/Dance_v2/LODGE/data/motorica/mofea")

