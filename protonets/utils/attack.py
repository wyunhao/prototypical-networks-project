import torch
import torch.nn.functional as F
import random
from protonets.utils.metric import calculate_loss_metric

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def attack_pgd(embedding_net, config, attacked_on_data, data_support_proto, original_labels, n_way, n_query, n_shot):
    if config['targeted']:
        new_labels = torch.zeros_like(original_labels)
        for i in range(int(original_labels.size()[0])):
            for j in range(int(original_labels.size()[1])):
                if j == 0: # make all support label targeting the same class
                    new_labels[i,j] = new_labels[i,0]
                while True and j != 0:
                    new_labels[i,j] = random.randint(0,n_way-1)
                    if new_labels[i,j] != original_labels[i,j]:
                        break
    else:
        new_labels = original_labels

    print("       check new label: ", new_labels)

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
