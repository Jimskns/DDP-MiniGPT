import torch
from torch.utils.data import Dataset
import fsspec
from config import DataConfig

class CharDataset(Dataset):
    def __init__(self, config:DataConfig) -> None:
        super().__init__()
        data = fsspec.open(config.path).open().read().decode('utf-8')
        data = data[: int(len(data) * config.truncate)]
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print("Data has %d charaters, %d unique." % (data_size, vocab_size))
        self.stoi = {ch : i for i, ch in enumerate(chars)}
        self.itos = {i : ch for i, ch in enumerate(chars)}
        self.block_size = config.block_size
        self.vocab_size = vocab_size
        self.data = data
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.block_size+1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
