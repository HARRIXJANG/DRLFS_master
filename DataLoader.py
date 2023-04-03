from dgl.data import DGLDataset
import torch

class GraphDataset(DGLDataset):
    def __init__(self, Graphs,  raw_dir=None, force_reload=False, verbose=False):
        self.Graphs = Graphs
        super(GraphDataset, self).__init__(name='Costdata',
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)

    def process(self):
        self.graphs = []
        for i in range(self.Graphs.shape[0]):
            TempGraph = self.Graphs[i]
            self.graphs.append(TempGraph)
    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)
