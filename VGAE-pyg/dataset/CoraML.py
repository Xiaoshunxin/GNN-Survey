import torch
from torch_geometric.data import InMemoryDataset, download_url
from utils.cora_ml import read_cora_ml_data

class CoraML(InMemoryDataset):

    url = "https://github.com/abojchevski/graph2gauss/tree/master/data"

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(CoraML, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = ['cora_ml.npz']
        return names

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        data = read_cora_ml_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)



















