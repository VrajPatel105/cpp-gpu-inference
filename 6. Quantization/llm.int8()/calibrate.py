import torch

running_max = {}

def make_hook(name):
    def hook_func(module, input, output):
        x = input[0] # the input is just (tensor,) but pytorch func requires this to be a tuple.
        # lets get the named vars just for more simplicity
        batch = x.shape[0]
        seq_len = x.shape[1]
        hidden_dim = x.shape[2]
        # first reshape 
        x = x.reshape(batch*seq_len, hidden_dim)
        # now convert the values in hidden_dim to abs (meaning to simply apply modulus)
        x = x.abs()
        # now get the max on hidden_dim col (channel)
        x = x.max(dim=0).values

        # this finally means that we have per channel max for that tensor (channel aka hidden_dim)

        # now updating the running max
        if name not in running_max:
            running_max[name] = x 
        else:
            running_max[name] = torch.maximum(running_max[name], x)
    return hook_func