import torch
import torch.nn.functional as F
import random
from protonets.utils.metric import calculate_loss_metric

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
