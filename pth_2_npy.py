import numpy as np

NAS_BENCH_201 = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
pth = "datasets/nasbench201/NAS-Bench-201-v1_1-096897.pth"
from nas_201_api.api_201 import NASBench201API

def nasbench201_to_nasbench101(arch_list):
    num_ops = sum(range(1, 1 + len(arch_list))) + 2
    adj = np.zeros((num_ops, num_ops), dtype=np.uint8)
    ops = ['input', 'output']
    node_lists = [[0]]
    for node_201 in arch_list:
        node_list = []
        for node in node_201:
            node_idx = len(ops) - 1
            adj[node_lists[node[1]], node_idx] = 1
            ops.insert(-1, node[0])
            node_list.append(node_idx)
        node_lists.append(node_list)
    adj[-(1+len(arch_list)):-1, -1] = 1
    arch = {'adj': adj,
            'ops': ops,}
    return arch


def extract(nasbench,arch):

    cifar10_train=  nasbench.get_more_info(arch,'cifar10',hp=200,is_random=False)['train-accuracy']
    cifar10_valid = nasbench.get_more_info(arch, 'cifar10-valid',hp=200,is_random=False)['valid-accuracy']
    cifar10_test= nasbench.get_more_info(arch, 'cifar10',hp=200,is_random=False)['test-accuracy']
    cifar100_train=nasbench.get_more_info(arch, 'cifar100',hp=200,is_random=False)['train-accuracy']
    cifar100_valid = nasbench.get_more_info(arch, 'cifar100',hp=200,is_random=False)['valid-accuracy']
    cifar100_test = nasbench.get_more_info(arch, 'cifar100',hp=200,is_random=False)['test-accuracy']
    imagenet16_train=nasbench.get_more_info(arch, 'ImageNet16-120',hp=200,is_random=False)['train-accuracy']
    imagenet16_valid=nasbench.get_more_info(arch, 'ImageNet16-120',hp=200,is_random=False)['valid-accuracy']
    imagenet16_test=nasbench.get_more_info(arch, 'ImageNet16-120',hp=200,is_random=False)['test-accuracy']

    return cifar10_train, cifar10_valid, cifar10_test, cifar100_train, cifar100_valid, \
           cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test


if __name__ == '__main__':
    nasbench201_dict = {}
    arch_counter = 0
    nasbench = NASBench201API(pth, verbose=False)
    for archi in nasbench:
        cifar10_train, cifar10_valid, cifar10_test, cifar100_train, cifar100_valid, \
        cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test = extract(nasbench,archi)
        model_dict = {
            'arch': nasbench.str2lists(archi),
            'cifar10_train': cifar10_train,
            'cifar10_valid': cifar10_valid,
            'cifar10_test': cifar10_test,
            'cifar100_train': cifar100_train,
            'cifar100_valid': cifar100_valid,
            'cifar100_test': cifar100_test,
            'imagenet16_train': imagenet16_train,
            'imagenet16_valid': imagenet16_valid,
            'imagenet16_test': imagenet16_test,
        }
        nasbench201_dict.update({str(arch_counter):model_dict})
        arch_counter += 1

        if arch_counter % 1000 == 0:

            print(arch_counter)

        np.save('./datasets/nasbench201/nasbench201_dict_search.npy', nasbench201_dict)

    #exit(0)
    nasbench201_dict = np.load('./datasets/nasbench201/nasbench201_dict_search.npy', allow_pickle=True).item()
    for key in nasbench201_dict.keys():
        print(key)
        print(nasbench201_dict[key])

