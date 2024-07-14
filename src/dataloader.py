import os
import json

class Dataloader:
    def __init__(self, datasets, ROOT):
        self.datasets = datasets
        self.ROOT = ROOT

    def load_data(self):
        data = []
        for dataset in self.datasets:
            with open(os.path.join(self.ROOT, dataset), 'r') as f:
                data.append(json.load(f))
        return data