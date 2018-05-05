import torch


def get_cuda_if_available():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        capable = torch.cuda.get_device_capability(0)[0] >= 4
        if capable:
            return (True, device)
    return (False, torch.device('cpu'))
