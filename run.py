import os
import pickle
import torch
import pandas as pd
from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import ProteinDrugDataset
from model import ProteinDrugLLMModel
from train import train_model
import sys
import random
import numpy as np
import h5py
import argparse

os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--flag", type=str, choices=["local"], help="Use Google Drive paths if flag is 'local'")
args = parser.parse_args()

colab = args.flag == "local" 

if colab:
    h5_dir = '/content/drive/MyDrive/Shared-with-booknerd'
    csv_file = '/content/drive/MyDrive/Shared-with-booknerd/tier1_smiles_smol.csv'
    smiles_cache_file = '/content/drive/MyDrive/Shared-with-booknerd/smiles_cache.pkl'
    model_name = 'meta-llama/Llama-2-7b-hf'
    gnn_ckpt = '/content/drive/MyDrive/Shared-with-booknerd/gcn_contextpred.pth'
else:
    h5_dir = '/mnt/data/gene_data'
    csv_file = '/mnt/data/tier1.csv'
    smiles_cache_file = '/mnt/data/smiles_cache.pkl'
    model_name = '/mnt/data/vicuna-13b-v1.5'
    gnn_ckpt = '/mnt/data/gcn_contextpred.pth'

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["[PROTEIN]", "[DRUG]"]})
print("Added special tokens")

print("Loading full CSV and SMILES cache")
full_csv_df = pd.read_csv(csv_file)
with open(smiles_cache_file, 'rb') as f:
    smiles_cache = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ProteinDrugLLMModel(
    llm_model_name=model_name,
    tokenizer=tokenizer,
    gnn_ckpt=gnn_ckpt,
    freeze_gnn=True
)

h5_files = sorted([f for f in os.listdir(h5_dir) if f.endswith('.h5')])
for epoch in range(1, 5):
    print(f"\n=== Epoch {epoch} ===")
    for h5_fname in h5_files:
        h5_path = os.path.join(h5_dir, h5_fname)
        print(f"\n--- Training on stage: {h5_fname} ---")

        with h5py.File(h5_path, 'r') as h5_file:
            keys = set(h5_file.keys())
        relevant_rows = []
        for idx, row in full_csv_df.iterrows():
            gene_id = str(int(row['GeneID']))
            h5_key = f"genes_{gene_id}"
            if h5_key in keys and row['SMILES'] in smiles_cache:
                relevant_rows.append(row)
        csv_subset = pd.DataFrame(relevant_rows).reset_index(drop=True)

        dataset = ProteinDrugDataset(
            h5_file=h5_path,
            csv_subset=csv_subset,
            smiles_cache=smiles_cache,
            tokenizer=tokenizer
        )

        indices = list(range(len(dataset)))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=epoch)
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)

        trained_model = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            batch_size=2,
            log_predictions=True,
            log_frequency=5,
            num_epochs=1
        )

        # Optional: save checkpoint per stage
        torch.save(model.state_dict(), f"/mnt/data/checkpoints/model_epoch{epoch}_{h5_fname}.pt")