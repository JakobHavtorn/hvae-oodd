from torch.utils.data import Dataset


class BaseDataset(Dataset):
    _repr_attributes = []

    def __init__(self):
        super().__init__()

    def __getitem__(self, idx):
        raise NotImplementedError()

    def __len__(self):
        if not hasattr(self, 'examples'):
            raise NotImplementedError()
        return len(self.examples)

    @property
    def size(self):
        x, y = self[0]
        y_size = y.shape if hasattr(y, 'shape') else tuple()
        return x.shape, y_size

    def __repr__(self):
        s = f'{self.__class__.__name__}('
        s += ', '.join([f'{attr}={getattr(self, attr)}' for attr in self._repr_attributes])
        return s + ')'

