import torch
import torch.nn.functional as F
import random
from protonets.utils.metric import calculate_loss_metric

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def attack_pgd_targeted(embedding_net, config, attacked_on_data, data_support_proto, original_labels, n_way, n_query, n_shot):
    simiar_proto_map = _calculate_similar_proto_pair(data_support_proto)

    if config['targeted']:
        new_labels = torch.zeros_like(original_labels)
        for i in range(int(original_labels.size()[0])):
            for j in range(int(original_labels.size()[1])):
                new_labels[i,j] = simiar_proto_map[i]
    else:
        new_labels = original_labels

    x = attacked_on_data.detach()
    if config['random_init']:
        x = x + torch.zeros_like(x).uniform_(-config['epsilon'], config['epsilon'])
        x = torch.min(torch.max(x, attacked_on_data - config['epsilon']), attacked_on_data + config['epsilon'])
        x = torch.clamp(x, 0.0, 255.0)

    for i in range(config['num_steps']):

        x.requires_grad_()
        with torch.enable_grad():
            emb_query_adv = embedding_net.encoder(x.reshape([-1] + list(x.shape[-3:]))).reshape(5*n_query, -1)
            loss, _ = calculate_loss_metric(n_way, n_query, new_labels, emb_query_adv, data_support_proto)

        grad = torch.autograd.grad(loss, [x])[0]

        if config['targeted']:
            x = x.detach() - config['step_size']*torch.sign(grad.detach())
        else:
            x = x.detach() + config['step_size']*torch.sign(grad.detach())

        x = torch.min(torch.max(x, attacked_on_data - config['epsilon']), attacked_on_data + config['epsilon'])
        x = torch.clamp(x, 0.0, 255.0)
    
    return x


def attack_pgd(embedding_net, config, attacked_on_data, data_support_proto, original_labels, n_way, n_query, n_shot):
    if config['targeted']:
        new_labels = torch.zeros_like(original_labels)
        for i in range(int(original_labels.size()[0])):
            for j in range(int(original_labels.size()[1])):
                if j != 0: # make all support label targeting the same class
                    new_labels[i,j] = new_labels[i,0]
                while True and j == 0:
                    new_labels[i,j] = random.randint(0,n_way-1)
                    if new_labels[i,j] != original_labels[i,j]:
                        break
    else:
        new_labels = original_labels

    x = attacked_on_data.detach()
    if config['random_init']:
        x = x + torch.zeros_like(x).uniform_(-config['epsilon'], config['epsilon'])
        x = torch.min(torch.max(x, attacked_on_data - config['epsilon']), attacked_on_data + config['epsilon'])
        x = torch.clamp(x, 0.0, 255.0)

    for i in range(config['num_steps']):

        x.requires_grad_()
        with torch.enable_grad():
            emb_query_adv = embedding_net.encoder(x.reshape([-1] + list(x.shape[-3:]))).reshape(5*n_query, -1)
            loss, _ = calculate_loss_metric(n_way, n_query, new_labels, emb_query_adv, data_support_proto)

        grad = torch.autograd.grad(loss, [x])[0]

        if config['targeted']:
            x = x.detach() - config['step_size']*torch.sign(grad.detach())
        else:
            x = x.detach() + config['step_size']*torch.sign(grad.detach())

        x = torch.min(torch.max(x, attacked_on_data - config['epsilon']), attacked_on_data + config['epsilon'])
        x = torch.clamp(x, 0.0, 255.0)
    
    return x


def _calculate_similar_proto_pair(proto_vector):
    similar_proto_map = [] # similar_proto_map[i] is the most similar proto class to class i

    for i in range(int(proto_vector.size()[0])):
        for j in range(int(proto_vector.size()[0])):
            dists = {}
            dists[j] = _calculate_eculian_dist(proto_vector[i], proto_vector[j])
            dists = dict(sorted(dists.items(), key=lambda item: item[1]))
            similar_proto_map.append(dists.keys[1]) # pick the second similar, since the first is itself

    return similar_proto_map

def _calculate_eculian_dist(v1, v2):
    return torch.cdist(v1, v2, p=2)

def support_set_attack(embedding_net, config, data_query, data_support_proto, labels_query, n_way, n_query, n_shot):
    new_labels_query = labels_query

    x = data_query.detach()
    emb_query = embedding_net.encoder(x.reshape([-1] + list(x.shape[-3:]))).reshape(5*n_query, -1)
    
    if config['random_init']:
        x = x + torch.zeros_like(x).uniform_(-config['epsilon'], config['epsilon'])
        x = torch.min(torch.max(x, data_query - config['epsilon']), data_query + config['epsilon'])
        x = torch.clamp(x, 0.0, 255.0)

    
    for i in range(config['num_steps']):

        x.requires_grad_()
        with torch.enable_grad():
            x_embedded = embedding_net.encoder(x)
            query_dim = x_embedded.size(-1)
            data_support_proto_adv = x_embedded[:(n_way * n_shot)].view(n_way, n_shot, query_dim).mean(1)
            #emb_query_adv = embedding_net.encoder(x.reshape([-1] + list(x.shape[-3:]))).reshape(5*n_query, -1)
            loss, _ = calculate_loss_metric(n_way, n_query, new_labels_query, emb_query, data_support_proto_adv)

        grad = torch.autograd.grad(loss, [x])[0]

        if config['targeted']:
            x = x.detach() - config['step_size']*torch.sign(grad.detach())
        else:
            x = x.detach() + config['step_size']*torch.sign(grad.detach())

        x = torch.min(torch.max(x, data_query - config['epsilon']), data_query + config['epsilon'])
        x = torch.clamp(x, 0.0, 255.0)
    
    return x