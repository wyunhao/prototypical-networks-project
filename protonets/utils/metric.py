import torch
import torch.nn.functional as F

from protonets.utils.distance_measurement import euclidean_dist

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
    return loss_val, {
        'loss': loss_val.item(), 'acc': acc_val.item(), 'y_hat': y_hat}
