import os
import argparse
import numpy as np
import pandas as pd
import torch

# https://github.com/facebookresearch/esm

def extract_esm2_embedding(input, model, batch_converter, device, repr_layers=-1):
    batch_labels, batch_strs, batch_tokens = batch_converter(input)
    with torch.no_grad():
        results = model(batch_tokens.to(device), repr_layers=[repr_layers], return_contacts=False)
    token_embeddings = results["representations"][repr_layers]
    sequence_embeddings = token_embeddings.mean(1)
    return sequence_embeddings.cpu().numpy()


def get_batch_embedding(save_folder, csv_file, batch_size, 
                        device='', model_name='esm2_t36_3B_UR50D', repr_layers=33):
    
    data = pd.read_csv(csv_file, usecols=[0,1,2]).values
    print(f'start -- {csv_file}')
    
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
    batch_converter = alphabet.get_batch_converter() 
    model.eval()
    
    if device == '':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    batch_num = len(data)//batch_size + 1    
    for idx in range(0, batch_num):
        save_name = os.path.join(save_folder, f'batch_{idx}.npz')
        if not os.path.exists(save_name):
            batch_data = data[idx*batch_size : (idx+1)*batch_size].astype(str)
            sequences = [(i[0],i[1]) for i in batch_data]
            X = extract_esm2_embedding(sequences, model, batch_converter, device, repr_layers)
            y = batch_data[:,2].astype(float)
            name = batch_data[:,0].astype(str)
            np.savez(save_name, X=X, y=y, n=name)
            print(f'finish {idx}/{batch_num} -- {csv_file}')
    torch.cuda.empty_cache()
    

def parse_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dms_folder", 
        type=str,
        default="",
        help="Path to DMS data folder"
    )
    
    parser.add_argument(
        "--dms_file", 
        type=str,
        default="SPG1_STRSG_Wu_2016.csv"
    )    
    
    parser.add_argument(
        "--embed_folder",
        type=str,
        default="./embeddings/ESM2_650M/",
        help="Path to save ESM2 embeddings"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="esm2_t33_650M_UR50D",
        choices=["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D"],
        help="ESM2 model variant"
    )
    
    parser.add_argument(
        "--repr_layers",
        type=int,
        default=33,
        help="Which layer to use for embeddings"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for processing"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default='',
        help="device for processing (default: '')"
    )
    
    parser.add_argument(
        "--gpu",
        type=str,
        default='6',
        help="GPU for processing"
    )    
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    
    if args.gpu and not args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    if args.dms_file:
        file_for_embed = [args.dms_file]
    else:
        file_for_embed = os.listdir(args.dms_folder)
        
    for file in file_for_embed:
        save_folder = os.path.join(args.embed_folder, file.split('_')[0])
        
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        csv_file = os.path.join(args.dms_folder, file)
        
        get_batch_embedding(save_folder, csv_file, args.batch_size, 
                    args.device, args.model_name, args.repr_layers)
    