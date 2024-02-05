from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch
import torch.optim as optim

def train(args):

    model = model_factory[args.model]()
    loss_function = ClassificationLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_loader = load_data("data/train", num_workers=0, batch_size=128)
    valid_loader = load_data("data/valid", num_workers=0, batch_size=128)

    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        average_loss = total_loss / len(train_loader)
        print('Average Training Loss: {:.6f}'.format(average_loss))

        # Validate the model
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in valid_loader:
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        val_accuracy = correct / total
        print('Validation Accuracy: {:.2f}%'.format(val_accuracy * 100))

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
