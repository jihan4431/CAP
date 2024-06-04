import h5py
import numpy as np
from nasbench import api
from tqdm import tqdm

full_tfrecord = "datasets/nasbench101/nasbench_full.tfrecord"
nasbench_hdf5 = "datasets/nasbench101/nasbench.hdf5"
ops_ids = {
    "input": -1,
    "output": -2,
    "conv3x3-bn-relu": 0,
    "conv1x1-bn-relu": 1,
    "maxpool3x3": 2
}

nasbench = api.NASBench(full_tfrecord)
metrics_list = []
operations_list = []
adjacency_list = []
trainable_parameters_list = []
hash_list = []
num_vertices_list = []

for hashval in tqdm(nasbench.hash_iterator()):
    rawdata, metrics = nasbench.get_metrics_from_hash(hashval)
    hash_list.append(hashval.encode())
    trainable_parameters_list.append(rawdata["trainable_parameters"])
    num_vertices = len(rawdata["module_operations"])
    num_vertices_list.append(num_vertices)
    assert num_vertices <= 7

    adjacency_padded = np.zeros((7, 7), dtype=np.int8)
    adjacency = np.array(rawdata["module_adjacency"], dtype=np.int8)
    adjacency_padded[:adjacency.shape[0], :adjacency.shape[1]] = adjacency
    adjacency_list.append(adjacency_padded)

    operations = np.array(list(map(lambda x: ops_ids[x], rawdata["module_operations"])), dtype=np.int8)
    operations_padded = np.zeros((7,), dtype=np.int8)
    operations_padded[:operations.shape[0]] = operations
    operations_list.append(operations_padded)

    metrics_list.append([])
    for epoch in [4, 12, 36, 108]:
        converted_metrics = []
        for idx in range(3):
            cur = metrics[epoch][idx]
            converted_metrics.append(np.array([[cur[pre + "_training_time"],
                                                cur[pre + "_train_accuracy"],
                                                cur[pre + "_validation_accuracy"],
                                                cur[pre + "_test_accuracy"]] for pre in ["halfway", "final"]
                                               ], dtype=np.float32))
        metrics_list[-1].append(converted_metrics)
hash_list = np.array(hash_list)
operations_list = np.stack(operations_list)
adjacency_list = np.stack(adjacency_list)
trainable_parameters_list = np.array(trainable_parameters_list, dtype=np.int32)
metrics_list = np.array(metrics_list, dtype=np.float32)
num_vertices_list = np.array(num_vertices_list, dtype=np.int8)

with h5py.File(nasbench_hdf5, "w") as fp:
    fp.create_dataset("hash", data=hash_list)
    fp.create_dataset("num_vertices", data=num_vertices_list)
    fp.create_dataset("trainable_parameters", data=trainable_parameters_list)
    fp.create_dataset("adjacency", data=adjacency_list)
    fp.create_dataset("operations", data=operations_list)
    fp.create_dataset("metrics", data=metrics_list)
