from torch.utils.data import Dataset

class IceDataset(Dataset):
    def __init__(self, x:dict, y:dict) -> None:
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x.keys())
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]