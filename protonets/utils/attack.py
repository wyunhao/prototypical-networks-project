import torch
import torch.nn.functional as F
import random
from protonets.utils.distance_measurement import euclidean_dist

from torch.autograd import Variable

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def calculate_loss_metric(num_way, num_query, target_inds, z_query, z_proto):
    dists = euclidean_dist(z_query, z_proto)

    # compute the log probabilities
    log_p_y = F.log_softmax(-dists, dim = 1).view(num_way, num_query, -1)

    # compute the loss
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

    # get the predicted labels for each query
    _, y_hat = log_p_y.max(2) # lines are classes and columns are query embeddings

    # compute the accuracy
    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

    # return output: loss, acc and predicted value
    return loss_val, {'loss': loss_val.item(), 'acc': acc_val.item(), 'y_hat': y_hat}


def _get_protonet_head(query, prototypes, n_way, n_query, n_shot):
    """
    Constructs the prototype representation of each class(=mean of support vectors of each class) and 
    returns the classification score (=L2 distance to each class prototype) on the query set.
    
    This model is the classification head described in:
    Prototypical Networks for Few-shot Learning
    (Snell et al., NIPS 2017).
    
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    tasks_per_batch = query.size(0)
    n_support = n_way * n_shot
    d = query.size(2)
    
    assert(query.size(0) == prototypes.size(0) and query.size(2) == prototypes.size(1))

    # Distance Matrix Vectorization Trick
    prototypes = torch.repeat_interleave(prototypes, n_query, dim=0).view(n_way, n_query, d)

    AB = _compute_gram_matrix(query, prototypes)
    AA = (query * query).sum(dim=2, keepdim=True)
    BB = (prototypes * prototypes).sum(dim=2, keepdim=True)
    logits = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)

    return -logits / d


def attack_pgd(embedding_net, config, data_query, data_support_proto, labels_query, n_way, n_query, n_shot):
    if config['targeted']:
        new_labels_query = torch.zeros_like(labels_query)
        for i in range(int(labels_query.size()[0])):
            for j in range(int(labels_query.size()[1])):
                while True:
                    new_labels_query[i,j] = random.randint(0,n_way-1)
                    if new_labels_query[i,j] != labels_query[i,j]:
                        break
    else:
        new_labels_query = labels_query

    x = data_query.detach()
    if config['random_init']:
        x = x + torch.zeros_like(x).uniform_(-config['epsilon'], config['epsilon'])
        x = torch.min(torch.max(x, data_query - config['epsilon']), data_query + config['epsilon'])
        x = torch.clamp(x, 0.0, 255.0)

    for i in range(config['num_steps']):

        x.requires_grad_()
        with torch.enable_grad():
            emb_query_adv = embedding_net.encoder(x.reshape([-1] + list(x.shape[-3:]))).reshape(5*n_query, -1)
            loss, _ = calculate_loss_metric(n_way, n_query, new_labels_query, emb_query_adv, data_support_proto)

        grad = torch.autograd.grad(loss, [x])[0]

        if config['targeted']:
            x = x.detach() - config['step_size']*torch.sign(grad.detach())
        else:
            x = x.detach() + config['step_size']*torch.sign(grad.detach())

        x = torch.min(torch.max(x, data_query - config['epsilon']), data_query + config['epsilon'])
        x = torch.clamp(x, 0.0, 255.0)
    
    return x


def _one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies


def _compute_gram_matrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.
    
    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """
    return torch.bmm(A, B.transpose(1,2))
