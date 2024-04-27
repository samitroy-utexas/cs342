import torch
import torch.nn as nn

class IceHockeyKartNetWithActions(nn.Module):

    def __init__(self):
        super(IceHockeyKartNetWithActions, self).__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(17, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
        )

        self.classifier = torch.nn.Linear(128, 3)

    def forward(self, x):
        x = x.squeeze(dim=1)
        x = self.network(x)
        x = self.classifier(x)
        acceleration = x[:, 0].unsqueeze(1)
        steer = x[:, 1].unsqueeze(1)
        brake = x[:, 2].unsqueeze(1)
        return acceleration, steer, brake


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, IceHockeyKartNetWithActions):
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, path.join(path.dirname(path.abspath(__file__)), 'action_net.pt'))
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'action_net.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = IceHockeyKartNetWithActions()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'action_net.th'), map_location='cpu'))
    return r