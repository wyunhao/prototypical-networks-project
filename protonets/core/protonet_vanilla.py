from warnings import filterwarnings
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.attack import support_set_attack

# ignore pytorch boring warnings
filterwarnings("ignore", category=UserWarning)

# the model below computes a class prototype by averaging the support embeddings of a class

# check if there is cuda available
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ProtoNet(nn.Module):
    def __init__(self, encoder):
        super(ProtoNet, self).__init__()
        self.encoder = encoder.to(dev)

    def set_forward_loss(self, episode_dict, attack_query=False, attack_support=False, attack_pgd=None, config=None):
        # extract all images
        images = episode_dict['images'].to(dev)

        # get episode setup
        num_way = episode_dict['num_way'] # way
        num_shot = episode_dict['num_shot'] # shot
        num_query = episode_dict['num_query'] # number of query images

        # from each class, extract num_shot support images
        x_support = images[:, :num_shot] # lines are classes and columns are images

        # from each class, extract the remaining images as query images
        x_query = images[:, num_shot:] # lines are classes and columns are images

        # create indices from 0 to num_way-1 for classification
        target_inds = torch.arange(0, num_way).view(num_way, 1, 1)

        # replicate all indices num_query times (for each query image)
        target_inds = target_inds.expand(num_way, num_query, 1).long()

        # convert indices from Tensor to Variable
        target_inds = Variable(target_inds, requires_grad = False).to(dev)

        # transform x_support into a array in which all images are contiguous
        x_support = x_support.contiguous().view(
            num_way * num_shot, *x_support.size()[2:]) # no more lines and columns

        # encode all images and compute class prototypes
        x_embedded = self.encoder.forward(x_support) # embeddings
        query_dim = x_embedded.size(-1)
        proto = x_embedded[:(num_way * num_shot)].view(num_way, num_shot, query_dim).mean(1)

        # transform x_query into a array in which all images are contiguous
        x_query = x_query.contiguous().view(
            num_way * num_query, *x_query.size()[2:]) # no more lines and columns

        # if attack the support, recalculate the proto based on the attacked support img for future querying
        if attack_support:
            support_label = torch.arange(0, num_way).view(num_way, 1, 1)
            support_label = support_label.expand(num_way, num_shot, 1).long()
            support_label = Variable(support_label, requires_grad = False).to(dev)

            #support_attack = support_set_attack(self, config, x_support, proto, support_label, num_way, num_shot, num_shot)
            support_attack = attack_pgd(self, config, x_support, proto, support_label, num_way, num_shot, num_shot)
            support_embedded = self.encoder.forward(support_attack)
            query_dim = support_embedded.size(-1)
            proto = support_embedded[:(num_way * num_shot)].view(num_way, num_shot, query_dim).mean(1)
        
        if attack_query:
            query_attack = attack_pgd(self, config, x_query, proto, target_inds, num_way, num_query, num_shot)
            query_embedded = self.encoder.forward(query_attack)
        else:
            query_embedded = self.encoder.forward(x_query)

        return num_way, num_query, target_inds, query_embedded, proto


# function to load the model structure
def load_protonet(x_dim, hid_dim, z_dim):
    # define a convolutional block
    def conv_block(layer_input, layer_output):
        conv = nn.Sequential(
            nn.Conv2d(layer_input, layer_output, 3, padding=1),
            nn.BatchNorm2d(layer_output), nn.ReLU(),
            nn.MaxPool2d(2))

        return conv

    # create the encoder to the embeddings for the images
    # the encoder is made of four conv blocks
    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim), conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim), conv_block(hid_dim, z_dim), Flatten())

    return ProtoNet(encoder)
