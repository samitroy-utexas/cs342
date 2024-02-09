from torch import optim

from .models import CNNClassifier, save_model, ClassificationLoss
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = CNNClassifier()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    loss_function = ClassificationLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_loader = load_data("data/train", num_workers=0, batch_size=128)
    valid_loader = load_data("data/valid", num_workers=0, batch_size=128)

    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().numpy()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        average_loss = total_loss / len(train_loader)
        print('Average Training Loss: {:.6f}'.format(average_loss))
        train_logger.add_scalar('loss', average_loss, global_step=epoch)

        # Validate the model
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().detach().cpu().numpy()

        val_accuracy = correct / total
        print('Validation Accuracy: {:.2f}%'.format(val_accuracy * 100))
        valid_logger.add_scalar('accuracy', val_accuracy, global_step=epoch)

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
