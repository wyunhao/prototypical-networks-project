from os import path
from protonets.utils.attack import attack_pgd
from protonets.utils.metric import calculate_loss_metric
from tqdm import trange
from math import fsum


import torch
from torch.autograd import Variable

import json

from protonets.core.episode_extractor import extract_episode
from protonets.core.dataset_loader import load_images
from protonets.core.model_loader import load_model

from protonets.utils.yaml_loader import load_yaml
from protonets.utils.logging import get_logger
from protonets.utils.time_measurement import measure_time


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# function to evaluate the model on test set
def evaluate_test(model, opt, test_data, logger):
    # load the saved model
    state_dict = torch.load(path.join(opt['results_dir'], 'best_model.pth'))
    model.load_state_dict(state_dict)

    # set the model to evaluation mode
    model.eval()

    test_loss = 0.0
    test_acc = []

    logger.info('> Testing')

    config = {
        'epsilon': 8.0/255.0,
        'num_steps': 7,
        'step_size': 2.0/255.0,
        'targeted': True,
        'random_init': True
    }

    # do epoch_size classification tasks to test the model
    for _ in trange(test_data['epoch_size']):
        # get the episode_dict
        episode_dict = extract_episode(
            test_data['test_x'], test_data['test_y'], test_data['num_way'],
            test_data['num_shot'], test_data['num_query'])

        # classify images and get the loss and the acc of the curr episode
        num_way, num_query, labels_query, _, z_proto, x_query = model.set_forward_loss(episode_dict)
        
        # for all query data image, perform PGD to get the perturbation
        x_query_attack = attack_pgd(model, config, x_query, z_proto, labels_query, num_way, num_query, test_data['num_shot'])

        # and then feed into the encoder to get the mapping onto its feature space
        z_query_attack = model.encoder.forward(x_query_attack)

        # caculate the loss of this perturbation
        _, output = calculate_loss_metric(num_way, num_query, labels_query, z_query_attack, z_proto)

        # acumulate the loss and the acc
        test_loss += output['loss']
        test_acc.append(output['acc'])

    # average the loss
    test_loss = test_loss / test_data['epoch_size']

    # average the acc
    test_acc_avg = sum(test_acc) / test_data['epoch_size']

    # calculate the standard deviation
    test_acc_dev = fsum([((x - test_acc_avg) ** 2) for x in test_acc])
    test_acc_dev = (test_acc_dev / (test_data['epoch_size'] - 1)) ** 0.5

    # calculate error considering 95% confidence interval
    error = 1.96 * test_acc_dev / (test_data['epoch_size'] ** 0.5)

    # output the test loss and the test acc
    logger.info('Loss: %.4f / Acc: %.2f +/- %.2f%%' % (test_loss, test_acc_avg * 100, error * 100))

    return test_acc_avg


def _generate_support_label(num_way, num_shot):
    # create indices from 0 to num_way-1 for classification
    target_inds = torch.arange(0, num_way).view(num_way, 1, 1)

    # replicate all indices num_shot times (for each support image)
    target_inds = target_inds.expand(num_way, num_shot, 1).long()

    # convert indices from Tensor to Variable
    target_inds = Variable(target_inds, requires_grad = False).to(dev)


# function to run evaluation n times
def evaluate_n_times(n, *args):
    test_acc_list = []

    test_acc = 0
    std_dev = 0

    for i in range(n):
        output = evaluate_test(*args)

        test_acc_list.append(output)
        test_acc += output

    # standard deviation
    test_acc = test_acc / n

    # standard deviation
    std_dev = fsum([((x - test_acc) ** 2) for x in test_acc_list])
    std_dev = (std_dev / (n - 1)) ** 0.5

    # calculate error considering 95% confidence interval
    error = 1.96 * std_dev / (n ** 0.5)

    # output the test loss and the test acc
    args[3].info('With %i run(s), Acc: %.2f +/- %.2f%%' % (n, test_acc * 100, error * 100))


# let's evaluate the model
def main():
    # get the path to the script from the current working directory
    script_path = path.dirname(__file__)

    # read the config file
    config = load_yaml(path.join(script_path, 'config', 'config.yaml'))

    # create a opt dict
    opt = {}

    opt.update(config['parameters'])

    # add directories to opt dict
    directories = {
        'data_dir': path.join(script_path, 'datasets'),
        'results_dir': path.join(script_path, 'results'),
        'logging_dir': path.join(script_path, 'results', 'logs')}

    opt.update(directories)

    # recover the chosen model and the dataset
    with open(path.join(opt['results_dir'], 'info.json'), 'r', encoding='utf8') as f:
        info_dict = json.load(f)

        model_name = info_dict['model']
        dataset = info_dict['dataset']

    # load the desired model
    if model_name == 'random_weights':
        weights_path = path.join(opt['results_dir'], 'weights.pkl')
        model = load_model(model_name, (3, 84, 84), 64, 64, weights_path)
    elif model_name == 'vanilla':
        model = load_model(model_name, (3, 84, 84), 64, 64)

    # create test_data dict
    test_data = config[dataset]['test']

    # load test set
    dataset_dir = path.join(opt['data_dir'], dataset)

    test_x, test_y = load_images(path.join(dataset_dir, 'test.pkl'))

    # add test set to test_data
    test_data.update({
        'test_x': test_x,
        'test_y': test_y})

    # configure the logging instance
    logger = get_logger(opt['logging_dir'], 'test.log')

    # run evaluation on test 15 times
    time_taken = measure_time(evaluate_n_times, 15, model, opt, test_data, logger)

    # output the time taken to test
    logger.info('Time taken by the test: %s seconds' % str(time_taken))


if __name__ == '__main__':
    main()
