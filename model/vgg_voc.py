import torch
import torch.nn as nn
from model.vgg_deeplab import Vgg_Deeplab as Deeplab
from model.util import *

sys.path.append('../')
import config


class Projection(nn.Module):

    def __init__(self, strong_or_weak, which_embedding, split):
        super(Projection, self).__init__()
        self.which_embedding = which_embedding
        self.strong_or_weak = strong_or_weak
        self.hidden = 300
        self.split = split
        if which_embedding == "all":
            self.hidden = 600
        self.projection = nn.Conv2d(1024, self.hidden, 1)
        self.strong_W, self.weak_W, self.all_W = self._get_W()

        nn.init.kaiming_normal_(self.projection.weight, a=1)
        nn.init.constant_(self.projection.bias, 0)

    def forward(self, x, which_W=None):
        if not which_W:
            which_W = self.strong_or_weak
        x = self.projection(x)
        x = x.permute([0, 2, 3, 1])
        assert which_W in ["strong", "weak", "all"]
        if which_W == "strong":
            x = torch.matmul(x, self.strong_W)
        elif which_W == "weak":
            x = torch.matmul(x, self.weak_W)
        elif which_W == "all":
            x = torch.matmul(x, self.all_W)
        x = x.permute([0, 3, 1, 2])
        return x

    def _get_W(self):
        embeddings = get_embeddings()
        Ws = get_Ws_split(embeddings, self.split)
        string = self.which_embedding + "_strong"
        strong = torch.tensor(Ws[string].T, dtype=torch.float).to(config.DEVICE)
        string = self.which_embedding + "_weak"
        weak = torch.tensor(Ws[string].T, dtype=torch.float).to(config.DEVICE)#.cuda()
        string = self.which_embedding + "_all"
        all = torch.tensor(Ws[string].T, dtype=torch.float).to(config.DEVICE)#.cuda()

        return strong, weak, all

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.projection.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{"params": self.get_10x_lr_params(), "lr": 10 * args.learning_rate}]



class Our_Model(nn.Module):
    """
    The main model of our algorithm
    """
    def __init__(self, split):
        super(Our_Model, self).__init__()
        self.vgg = Deeplab()
        self.projection = Projection("all", "all", split)

    def forward(self, x, which_W=None, which_branch=None):
        x = self.vgg(x)
        #print(x.shape)
        assert which_W in ["strong", "weak", "all", None]
        return self.projection(x, which_W)

    def get_1x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """

        for i in self.vgg.features:
            jj = 0
            for k in i.parameters():
                jj += 1
                if k.requires_grad:
                    yield k

    def optim_parameters_1x(self, args):
        return [{"params": self.get_1x_lr_params(), "lr": 1 * args.learning_rate}]

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.projection.projection.parameters())
        b.append(self.vgg.classifier.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters_10x(self, args):
        return [{"params": self.get_10x_lr_params(), "lr": 10 * args.learning_rate}]




class Decoder(nn.Module):
    def __init__(self, split):
        super(Decoder, self).__init__()
        self.encoder = Our_Model(split)
        self.decoder = {}

    def forward(self, x, which_W=None, which_branch=None):
        x = self.encoder(x)
        return x
