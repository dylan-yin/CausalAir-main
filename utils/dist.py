import pickle
import time

import torch
import torch.distributed as dist

def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

def all_gather_tensor(data):
    """
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # obtain Tensor size of each rank
    local_size = torch.tensor(data.shape).to(torch.device("cuda"))
    size_list = [local_size.clone() for _ in range(world_size)]
    dist.all_gather(size_list, local_size)

    tensor = data.view(-1)
    # obtain Tensor numel of each rank
    local_length = torch.LongTensor([tensor.numel()]).to(torch.device("cuda"))
    length_list = [torch.LongTensor([0]).to(torch.device("cuda")) for _ in range(world_size)]
    dist.all_gather(length_list, local_length)
    length_list = [int(length.item()) for length in length_list]
    max_length = max(length_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in length_list:
        tensor_list.append(torch.zeros((max_length,), dtype=tensor.dtype, device=tensor.device))
    if local_length != max_length:
        padding = torch.zeros((max_length-local_length,), dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for length, size, tensor in zip(length_list, size_list, tensor_list):
        buffer = tensor[:length].view(size)
        data_list.append(buffer)

    return data_list


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    if isinstance(data, torch.Tensor):
        is_tensor = True
        data = data.cpu().numpy()
    else:
        is_tensor = False

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(torch.cuda.current_device())

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to(torch.cuda.current_device())
    size_list = [torch.LongTensor([0]).to(torch.cuda.current_device()) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to(torch.cuda.current_device()))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to(torch.cuda.current_device())
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        tmp_data = torch.tensor(pickle.loads(buffer), device=torch.cuda.current_device()) if is_tensor else pickle.loads(buffer)
        data_list.append(tmp_data)

    return data_list

