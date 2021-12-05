import torch
import torch.nn.functional as F
import random

from torch.autograd import Variable

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
        #print(labels_query.shape)
        new_labels_query = torch.zeros_like(labels_query)
        for i in range(int(labels_query.size()[0])):
            for j in range(int(labels_query.size()[1])):
                while True:
                    new_labels_query[i,j] = random.randint(0,n_way-1)
                    if new_labels_query[i,j] != labels_query[i,j]:
                        break
    else:
        new_labels_query = labels_query

    new_labels_query = new_labels_query.view(new_labels_query.size()[0]*new_labels_query.size()[1])
    x = data_query.detach()
    #print(x.shape)
    if config['random_init']:
        x = x + torch.zeros_like(x).uniform_(-config['epsilon'], config['epsilon'])

    for i in range(config['num_steps']):

        x.requires_grad_()
        with torch.enable_grad():
            emb_query_adv = embedding_net.encoder(x.reshape([-1] + list(x.shape[-3:]))).reshape(5, n_query, -1)

            logits = _get_protonet_head(emb_query_adv, data_support_proto, n_way, n_query, n_shot)
            logits = logits.view(logits.size()[0]*logits.size()[1],logits.size()[2])

            loss = F.cross_entropy(logits, new_labels_query, size_average=False)
            #print(loss)

        grad = torch.autograd.grad(loss, [x])[0]
        #print(grad[0][0][0])
        if config['targeted']:
            x = x.detach() - config['step_size']*torch.sign(grad.detach())
        else:
            x = x.detach() + config['step_size']*torch.sign(grad.detach())

        x = torch.min(torch.max(x, data_query - config['epsilon']), data_query + config['epsilon'])
        x = torch.clamp(x, 0.0, 255.0)
        #print(x[0][0][10])
    
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
