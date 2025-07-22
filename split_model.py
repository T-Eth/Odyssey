import torch.nn as nn

def split_model_into_Sa_Sb(model: nn.Module):
    """
    Automatically split a model into Sa (convolutional layers) and Sb (fully connected layers).
    Assumes model has a conv + flatten + fc architecture.

    Returns:
        Sa (nn.Module): convolutional feature extractor
        Sb (nn.Module): classifier head
    """
    conv_layers = []
    fc_layers = []
    flatten_found = False

    for name, module in model.named_children():
        if not flatten_found:
            # Heuristic: once we find a Linear layer, assume the rest is Sb
            # and all previous are part of Sa
            if isinstance(module, nn.Sequential):
                sub_layers = list(module.children())
                for sub in sub_layers:
                    if isinstance(sub, nn.Linear):
                        flatten_found = True
                        fc_layers.append(sub)
                    elif flatten_found:
                        fc_layers.append(sub)
                    else:
                        conv_layers.append(sub)
            elif isinstance(module, nn.Linear):
                flatten_found = True
                fc_layers.append(module)
            else:
                conv_layers.append(module)
        else:
            fc_layers.append(module)

    # Wrap them in nn.Sequential modules
    Sa = nn.Sequential(*conv_layers)
    Sb = nn.Sequential(*fc_layers)
    return Sa, Sb