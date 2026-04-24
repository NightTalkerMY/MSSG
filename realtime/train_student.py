import torch
from torch.utils.data import Dataset, DataLoader

# This is a brand new, ultra-fast Dataset class
class CachedHoloPiDataset(Dataset):
    def __init__(self, cache_path):
        print(f"Loading cached tensors from {cache_path}...")
        data = torch.load(cache_path)
        self.v = data['v']
        self.a = data['a']
        self.t = data['t']
        self.l = data['l']
        print("Data loaded instantly!")

    def __len__(self):
        return self.v.shape[0]

    def __getitem__(self, idx):
        return self.v[idx], self.a[idx], self.t[idx], self.l[idx]

if __name__ == '__main__':
    # 1. Instantly load the cached data
    fast_dataset = CachedHoloPiDataset(r"D:\FYP\MSSG\predata\train_cache_50_buffer.pt")
    
    # 2. Because it's cached, you can safely use num_workers=4 on Windows without crashing!
    train_loader = DataLoader(fast_dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Test it
    for v_in, a_in, teacher_out, labels in train_loader:
        print("Successfully loaded a batch from the cache!")
        print(f"Visual Shape: {v_in.shape}")
        break
        
    # --- PHASE 2 to 5 (Neural Networks & Training Loop) WILL GO HERE ---