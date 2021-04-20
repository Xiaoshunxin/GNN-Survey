import numpy as np
import pdb
import scipy.sparse as sp
from sklearn.preprocessing import normalize


def preprocessor(domain="cora"):
    dataset = None
    if domain == "cora":
        dataset = load_cora_data()
    elif domain == "citeseer":
        dataset = load_citeseer_data()
    elif domain == "pubmed":
        dataset = load_pubmed_data()
    elif domain == "dblp":
        dataset = load_dblp_data()
    elif domain == "cora_ml":
        dataset = load_cora_ml_data()
    elif domain == "cora_large":
        dataset = load_cora_large_data()
    elif domain == "cs":
        dataset = load_cs_data()
    elif domain == "photo":
        dataset = load_photo_data()
    elif domain == "computers":
        dataset = load_computers_data()

    return dataset


def load_computers_data(file_name="./data/amazon_electronics_computers.npz"):
    print('Loading {} dataset...'.format("computers"))
    if not file_name.endswith('.npz'):
        file_name += '.npz'

    with np.load(file_name) as loader:
        loader = dict(loader)

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        adj_data = []
        adj_indices = []
        adj_indptr = [0]
        adj_shape = [13752, 13752]
        adj_indptr_source = loader['adj_indptr']
        adj_indices_source = loader['adj_indices']

        adj_matrix = np.zeros((13752, 13752), dtype=np.int)
        for i in range(1, len(adj_indptr_source)):
            left = adj_indptr_source[i - 1]
            right = adj_indptr_source[i]
            if left == right:
                continue
            indices = adj_indices_source[left:right]

            adj_matrix[i - 1, indices] = 1
            adj_matrix[indices, i - 1] = 1

        for i in range(len(adj_matrix)):
            row_connections = adj_matrix[i]
            count = 0
            for j in range(len(row_connections)):
                if row_connections[j] == 0:
                    continue
                else:
                    adj_data.append(row_connections[j])
                    adj_indices.append(j)
                    count += 1
            adj_indptr.append(adj_indptr[-1] + count)
        adj_data = np.array(adj_data, dtype=np.float32)
        A = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)

        """
        loader.get('labels')
            value: [0 1 1 ... 4 6 3]
            size: num_nodes
       """
        z = loader.get('labels')

        graph = {
            'A': A,
            'X': X,
            'z': z
        }
        return graph


def load_photo_data(file_name="./data/amazon_electronics_photo.npz"):
    print('Loading {} dataset...'.format("photo"))
    if not file_name.endswith('.npz'):
        file_name += '.npz'

    with np.load(file_name) as loader:
        loader = dict(loader)

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        adj_data = []
        adj_indices = []
        adj_indptr = [0]
        adj_shape = [7650, 7650]
        adj_indptr_source = loader['adj_indptr']
        adj_indices_source = loader['adj_indices']

        adj_matrix = np.zeros((7650, 7650), dtype=np.int)
        for i in range(1, len(adj_indptr_source)):
            left = adj_indptr_source[i - 1]
            right = adj_indptr_source[i]
            if left == right:
                continue
            indices = adj_indices_source[left:right]

            adj_matrix[i - 1, indices] = 1
            adj_matrix[indices, i - 1] = 1

        for i in range(len(adj_matrix)):
            row_connections = adj_matrix[i]
            count = 0
            for j in range(len(row_connections)):
                if row_connections[j] == 0:
                    continue
                else:
                    adj_data.append(row_connections[j])
                    adj_indices.append(j)
                    count += 1
            adj_indptr.append(adj_indptr[-1] + count)
        adj_data = np.array(adj_data, dtype=np.float32)
        A = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)

        """
        loader.get('labels')
            value: [0 1 1 ... 4 6 3]
            size: num_nodes
       """
        z = loader.get('labels')

        graph = {
            'A': A,
            'X': X,
            'z': z
        }
        return graph


def load_cs_data(file_name="./data/ms_academic_cs.npz"):
    print('Loading {} dataset...'.format("cs"))
    if not file_name.endswith('.npz'):
        file_name += '.npz'

    with np.load(file_name) as loader:
        loader = dict(loader)

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        adj_data = []
        adj_indices = []
        adj_indptr = [0]
        adj_shape = [18333, 18333]
        adj_indptr_source = loader['adj_indptr']
        adj_indices_source = loader['adj_indices']

        adj_matrix = np.zeros((18333, 18333), dtype=np.int)
        for i in range(1, len(adj_indptr_source)):
            left = adj_indptr_source[i - 1]
            right = adj_indptr_source[i]
            if left == right:
                continue
            indices = adj_indices_source[left:right]

            adj_matrix[i - 1, indices] = 1
            adj_matrix[indices, i - 1] = 1

        for i in range(len(adj_matrix)):
            row_connections = adj_matrix[i]
            count = 0
            for j in range(len(row_connections)):
                if row_connections[j] == 0:
                    continue
                else:
                    adj_data.append(row_connections[j])
                    adj_indices.append(j)
                    count += 1
            adj_indptr.append(adj_indptr[-1] + count)
        adj_data = np.array(adj_data, dtype=np.float32)
        A = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)

        """
        loader.get('labels')
            value: [0 1 1 ... 4 6 3]
            size: num_nodes
       """
        z = loader.get('labels')

        graph = {
            'A': A,
            'X': X,
            'z': z
        }
        return graph


def load_cora_large_data(file_name="./data/cora.npz"):
    print('Loading {} dataset...'.format("cora_large"))
    if not file_name.endswith('.npz'):
        file_name += '.npz'

    with np.load(file_name) as loader:
        loader = dict(loader)

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        a = loader['attr_shape']

        adj_data = []
        adj_indices = []
        adj_indptr = [0]
        adj_shape = [19793, 19793]
        adj_indptr_source = loader['adj_indptr']
        adj_indices_source = loader['adj_indices']

        adj_matrix = np.zeros((19793, 19793), dtype=np.int)
        for i in range(1, len(adj_indptr_source)):
            left = adj_indptr_source[i-1]
            right = adj_indptr_source[i]
            if left == right:
                continue
            indices = adj_indices_source[left:right]

            adj_matrix[i-1, indices] = 1
            adj_matrix[indices, i-1] = 1

        for i in range(len(adj_matrix)):
            row_connections = adj_matrix[i]
            count = 0
            for j in range(len(row_connections)):
                if row_connections[j] == 0:
                    continue
                else:
                    adj_data.append(row_connections[j])
                    adj_indices.append(j)
                    count += 1
            adj_indptr.append(adj_indptr[-1] + count)
        adj_data = np.array(adj_data, dtype=np.float32)
        A = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)

        """
        loader.get('labels')
            value: [0 1 1 ... 4 6 3]
            size: num_nodes
       """
        z = loader.get('labels')

        graph = {
            'A': A,
            'X': X,
            'z': z
        }
        return graph


def load_cora_ml_data(file_name="./data/cora_ml.npz"):
    print('Loading {} dataset...'.format("cora_ml"))
    if not file_name.endswith('.npz'):
        file_name += '.npz'

    with np.load(file_name) as loader:
        loader = dict(loader)

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        a = loader['attr_shape']

        adj_data = []
        adj_indices = []
        adj_indptr = [0]
        adj_shape = [19793, 19793]
        adj_indptr_source = loader['adj_indptr']
        adj_indices_source = loader['adj_indices']

        adj_matrix = np.zeros((19793, 19793), dtype=np.int)
        for i in range(1, len(adj_indptr_source)):
            left = adj_indptr_source[i-1]
            right = adj_indptr_source[i]
            if left == right:
                continue
            indices = adj_indices_source[left:right]

            adj_matrix[i-1, indices] = 1
            adj_matrix[indices, i-1] = 1

        for i in range(len(adj_matrix)):
            row_connections = adj_matrix[i]
            count = 0
            for j in range(len(row_connections)):
                if row_connections[j] == 0:
                    continue
                else:
                    adj_data.append(row_connections[j])
                    adj_indices.append(j)
                    count += 1
            adj_indptr.append(adj_indptr[-1] + count)
        adj_data = np.array(adj_data, dtype=np.float32)
        A = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)

        """
        loader.get('labels')
            value: [0 1 1 ... 4 6 3]
            size: num_nodes
       """
        z = loader.get('labels')

        graph = {
            'A': A,
            'X': X,
            'z': z
        }
        return graph


def load_dblp_data(file_name="./data/dblp.npz"):
    print('Loading {} dataset...'.format("dblp"))
    if not file_name.endswith('.npz'):
        file_name += '.npz'

    with np.load(file_name) as loader:
        loader = dict(loader)

        """
            loader['attr_data']
                type: numpy.ndarray
                value: [0.25524727 0.57424884 0.41145328 ... 0.34304865 0.2042381  0.48243084]
                size: the sum of all num_features (92192,)
            loader['attr_indices']
                type: numpy.ndarray
                value: [ 116  496  550 ... 1171 1571 1605]
                size: the sum of all num_features (92192,)
            loader['attr_indptr']
                type: numpy.ndarray
                value: [    0     5    12 ... 92180 92185 92192]
                shape: (num_nodes+1,)
                size: 17717
            loader['attr_shape']
                type: numpy.ndarray
                value: [17716  1639]
                shape: (2,)
                size: num_nodes * num_features
        """
        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])


        adj_data = []
        adj_indices = []
        adj_indptr = [0]
        adj_shape = [17716, 17716]
        adj_indptr_source = loader['adj_indptr']
        adj_indices_source = loader['adj_indices']

        adj_matrix = np.zeros((17716, 17716), dtype=np.int)
        for i in range(1, len(adj_indptr_source)):
            left = adj_indptr_source[i-1]
            right = adj_indptr_source[i]
            if left == right:
                continue
            indices = adj_indices_source[left:right]

            adj_matrix[i-1, indices] = 1
            adj_matrix[indices, i-1] = 1

        for i in range(len(adj_matrix)):
            row_connections = adj_matrix[i]
            count = 0
            for j in range(len(row_connections)):
                if row_connections[j] == 0:
                    continue
                else:
                    adj_data.append(row_connections[j])
                    adj_indices.append(j)
                    count += 1
            adj_indptr.append(adj_indptr[-1] + count)

        adj_data = np.array(adj_data, dtype=np.float32)
        A = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)

        """
        loader.get('labels')
            value: [0 1 1 ... 4 6 3]
            size: num_nodes
       """
        z = loader.get('labels')

        graph = {
            'A': A,
            'X': X,
            'z': z
        }
        return graph


def load_pubmed_data(path="./data2/pubmed/", dataset="pubmed"):
    print('Loading {} dataset...'.format(dataset))

    """
        idx_features_labels
            type: type 'numpy.ndarray
            shape: (num_nodes, num_features + 2)
    """
    attr_data = []
    attr_indices = []
    attr_indptr = [0]
    attr_shape = [19717, 500]

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))  # node_numbers * (id + features + label)

    nodes_to_idx = {j: i for i, j in enumerate(idx_features_labels[:, 0])}

    features = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
    # features = normalize(features, axis=1, norm='l2')
    for i in range(len(features)):
        row_feature = features[i]
        count = 0
        for j in range(len(row_feature)):
            if row_feature[j] == 0.:
                continue
            else:
                attr_data.append(row_feature[j])
                attr_indices.append(j)
                count += 1
        attr_indptr.append(attr_indptr[-1] + count)
    attr_data = np.array(attr_data, dtype=np.float32)
    X = sp.csr_matrix((attr_data, attr_indices, attr_indptr), shape=attr_shape)

    adj_data = []
    adj_indices = []
    adj_indptr = [0]
    adj_shape = [19717, 19717]

    """
        edges_unordered
            type: numpy.ndarray
            shape: (num_edges, 2)
    """
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.dtype(str))
    adj_matrix = np.zeros((19717, 19717), dtype=np.int)
    for i in range(len(edges_unordered)):
        left, right = edges_unordered[i]
        if left == right:
            continue
        left = nodes_to_idx[left]
        right = nodes_to_idx[right]
        adj_matrix[left][right] = 1
        adj_matrix[right][left] = 1

    for i in range(len(adj_matrix)):
        row_connections = adj_matrix[i]
        count = 0
        for j in range(len(row_connections)):
            if row_connections[j] == 0:
                continue
            else:
                adj_data.append(row_connections[j])
                adj_indices.append(j)
                count += 1
        adj_indptr.append(adj_indptr[-1] + count)
    adj_data = np.array(adj_data, dtype=np.float32)
    A = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)

    z = []
    labels = idx_features_labels[:, -1]
    label_to_idx = {j: i for i, j in enumerate(list(set(idx_features_labels[:, -1])))}
    for i in range(len(idx_features_labels[:, -1])):
        z.append(label_to_idx[labels[i]])
    z = np.array(z, dtype=np.int)

    graph = {
        'A': A,
        'X': X,
        'z': z
    }
    return graph


def load_citeseer_data(path="./data2/citeseer/", dataset="citeseer"):
    print('Loading {} dataset...'.format(dataset))

    """
        idx_features_labels
            type: type 'numpy.ndarray
            shape: (num_nodes, num_features + 2)
    """
    attr_data = []
    attr_indices = []
    attr_indptr = [0]
    attr_shape = [3312, 3703]

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))  # node_numbers * (id + features + label)

    nodes_to_idx = {j: i for i, j in enumerate(idx_features_labels[:, 0])}

    features = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
    features = normalize(features, axis=1, norm='l2')
    for i in range(len(features)):
        row_feature = features[i]
        count = 0
        for j in range(len(row_feature)):
            if row_feature[j] == 0.:
                continue
            else:
                attr_data.append(row_feature[j])
                attr_indices.append(j)
                count += 1
        attr_indptr.append(attr_indptr[-1] + count)
    attr_data = np.array(attr_data, dtype=np.float32)
    X = sp.csr_matrix((attr_data, attr_indices, attr_indptr), shape=attr_shape)

    adj_data = []
    adj_indices = []
    adj_indptr = [0]
    adj_shape = [3312, 3312]

    """
        edges_unordered
            type: numpy.ndarray
            shape: (num_edges, 2)
    """
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.dtype(str))
    adj_matrix = np.zeros((3312, 3312), dtype=np.int)
    for i in range(len(edges_unordered)):
        left, right = edges_unordered[i]
        left = nodes_to_idx[left]
        right = nodes_to_idx[right]

        adj_matrix[left][right] = 1
        adj_matrix[right][left] = 1

    for i in range(len(adj_matrix)):
        row_connections = adj_matrix[i]
        count = 0
        for j in range(len(row_connections)):
            if row_connections[j] == 0:
                continue
            else:
                adj_data.append(row_connections[j])
                adj_indices.append(j)
                count += 1
        adj_indptr.append(adj_indptr[-1] + count)
    adj_data = np.array(adj_data, dtype=np.float32)
    A = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)

    z = []
    labels = idx_features_labels[:, -1]
    label_to_idx = {j: i for i, j in enumerate(list(set(idx_features_labels[:, -1])))}
    for i in range(len(idx_features_labels[:, -1])):
        z.append(label_to_idx[labels[i]])
    z = np.array(z, dtype=np.int)

    graph = {
        'A': A,
        'X': X,
        'z': z
    }
    return graph


def load_cora_data(path="./data2/cora/", dataset="cora"):
    print('Loading {} dataset...'.format(dataset))

    """
        idx_features_labels
            type: type 'numpy.ndarray
            shape: (num_nodes, num_features + 2)
    """
    attr_data = []
    attr_indices = []
    attr_indptr = [0]
    attr_shape = [2708, 1433]

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))  # node_numbers * (id + features + label)

    nodes_to_idx = {j: i for i, j in enumerate(idx_features_labels[:, 0])}

    features = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
    features = normalize(features, axis=1, norm='l2')
    for i in range(len(features)):
        row_feature = features[i]
        count = 0
        for j in range(len(row_feature)):
            if row_feature[j] == 0.:
                continue
            else:
                attr_data.append(row_feature[j])
                attr_indices.append(j)
                count += 1
        attr_indptr.append(attr_indptr[-1] + count)
    attr_data = np.array(attr_data, dtype=np.float32)
    X = sp.csr_matrix((attr_data, attr_indices, attr_indptr), shape=attr_shape)

    adj_data = []
    adj_indices = []
    adj_indptr = [0]
    adj_shape = [2708, 2708]

    """
        edges_unordered
            type: numpy.ndarray
            shape: (num_edges, 2)
    """
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    adj_matrix = np.zeros((2708, 2708), dtype=np.int)
    for i in range(len(edges_unordered)):
        left, right = edges_unordered[i]
        left = nodes_to_idx[str(left)]
        right = nodes_to_idx[str(right)]

        adj_matrix[left][right] = 1
        adj_matrix[right][left] = 1

    for i in range(len(adj_matrix)):
        row_connections = adj_matrix[i]
        count = 0
        for j in range(len(row_connections)):
            if row_connections[j] == 0:
                continue
            else:
                adj_data.append(row_connections[j])
                adj_indices.append(j)
                count += 1
        adj_indptr.append(adj_indptr[-1] + count)
    adj_data = np.array(adj_data, dtype=np.float32)
    A = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)

    z = []
    labels = idx_features_labels[:, -1]
    label_to_idx = {j: i for i, j in enumerate(list(set(idx_features_labels[:, -1])))}
    for i in range(len(idx_features_labels[:, -1])):
        z.append(label_to_idx[labels[i]])
    z = np.array(z, dtype=np.int)

    graph = {
        'A': A,
        'X': X,
        'z': z
    }
    return graph



if __name__ == '__main__':
    load_cora_ml_data()


