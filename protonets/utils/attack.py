import torch
import torch.nn.functional as F
import random

from protonets.utils.metric import calculate_loss_metric


def attack_pgd(attack, embedding_net, config, data_query, emb_support, labels_query, labels_support, way, episodes_per_batch, n_query):
    if not attack:
        return data_query
    if config['targeted']:
        new_labels_query = torch.zeros_like(labels_query)
        for i in range(int(labels_query.size()[0])):
            for j in range(int(labels_query.size()[1])):
                while True:
                    new_labels_query[i,j] = random.randint(0,way-1)
                    if new_labels_query[i,j] != labels_query[i,j]:
                        break
    else:
        new_labels_query = labels_query
    new_labels_query = new_labels_query.view(new_labels_query.size()[0]*new_labels_query.size()[1])
    x = data_query.detach()
    if config['random_init']:
        x = x + torch.zeros_like(x).uniform_(-config['epsilon'], config['epsilon'])
    for i in range(config['num_steps']):
        x.requires_grad_()
        with torch.enable_grad():
            emb_query_adv = embedding_net.encoder(x.reshape([-1] + list(x.shape[-3:]))).reshape(episodes_per_batch, n_query, -1)

            logits = calculate_loss_metric(way, 1, labels_support, emb_query_adv, emb_support)
            
            loss = logits['loss']
        grad = torch.autograd.grad(loss, [x])[0]
        if config['targeted']:
            x = x.detach() - config['step_size']*torch.sign(grad.detach())
        else:
            x = x.detach() + config['step_size']*torch.sign(grad.detach())
        x = torch.min(torch.max(x, data_query - config['epsilon']), data_query + config['epsilon'])
        x = torch.clamp(x, 0.0, 1.0)
    return x