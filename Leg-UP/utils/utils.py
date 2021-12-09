import random

import numpy as np
import torch
from scipy import sparse

EPSILON = 1e-12
_fixed_target_items = {
    "head": np.asarray([259, 2272, 3010, 6737, 7690]),
    "tail": np.asarray([5611, 9213, 10359, 10395, 12308]),
    "upper_torso": np.asarray([1181, 1200, 2725, 4228, 6688]),
    "lower_torso": np.asarray([3227, 5810, 7402, 9272, 10551])
}


def sample_target_items(train_data, n_samples, popularity, use_fix=False):
    """Sample target items with certain popularity."""
    if popularity not in ["head", "upper_torso", "lower_torso", "tail"]:
        raise ValueError("Unknown popularity type {}.".format(popularity))

    n_items = train_data.shape[1]  # 14007
    all_items = np.arange(n_items)  # [0, 1, 2, ... , 14006]
    item_clicks = train_data.toarray().sum(0)

    valid_items = []
    if use_fix:
        valid_items = _fixed_target_items[popularity]
    else:
        bound_head = np.percentile(item_clicks, 95)
        bound_torso = np.percentile(item_clicks, 75)
        bound_tail = np.percentile(item_clicks, 50)
        if popularity == "head":
            valid_items = all_items[item_clicks > bound_head]
        elif popularity == "tail":
            valid_items = all_items[item_clicks < bound_tail]
        elif popularity == "upper_torso":
            valid_items = all_items[(item_clicks > bound_torso) & (item_clicks < bound_head)]
        elif popularity == "lower_torso":
            valid_items = all_items[(item_clicks > bound_tail) & (item_clicks < bound_torso)]

    if len(valid_items) < n_samples:
        raise ValueError("Cannot sample enough items that meet criteria.")

    np.random.shuffle(valid_items)
    sampled_items = valid_items[:n_samples]
    sampled_items.sort()
    print("Sampled target items: {}".format(sampled_items.tolist()))

    return sampled_items


def set_seed(seed, cuda=False):
    """Set seed globally."""
    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.manual_seed(seed)


def minibatch(*tensors, **kwargs):
    """Mini-batch generator for pytorch tensor."""
    batch_size = kwargs.get('batch_size', 128)  # 2048

    if len(tensors) == 1:  # √
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):  # len(tensor) = 14007
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    """Shuffle arrays."""
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def sparse2tensor(sparse_data):
    """Convert sparse csr matrix to pytorch tensor."""
    return torch.FloatTensor(sparse_data.toarray())


def tensor2sparse(tensor):
    """Convert pytorch tensor to sparse csr matrix."""
    return sparse.csr_matrix(tensor.detach().cpu().numpy())


def stack_csrdata(data1, data2):
    """Stack two sparse csr matrix."""
    return sparse.vstack((data1, data2), format="csr")


def save_fake_data(fake_data, path):
    """Save fake data to file."""
    file_path = "%s.npz" % path
    print("Saving fake data to {}".format(file_path))
    sparse.save_npz(file_path, fake_data)
    return file_path


def load_fake_data(file_path):
    """Load fake data from file."""
    fake_data = sparse.load_npz(file_path)
    print("Loaded fake data from {}".format(file_path))
    return fake_data


def save_checkpoint(model, optimizer, path, epoch=-1):
    """Save model checkpoint and optimizer state to file."""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    file_path = "%s.pt" % path
    print("Saving checkpoint to {}".format(file_path))
    torch.save(state, file_path)


def load_checkpoint(path):
    """Load model checkpoint and optimizer state from file."""
    file_path = "%s.pt" % path
    state = torch.load(file_path, map_location=torch.device('cpu'))
    print("Loaded checkpoint from {} (epoch {})".format(
        file_path, state["epoch"]))
    return state["epoch"], state["state_dict"], state["optimizer"]
