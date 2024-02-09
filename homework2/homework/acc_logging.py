from os import path
import torch
import torch.utils.tensorboard as tb


def test_logging(train_logger, valid_logger):

    global_step = 0  # Initialize global step

    # This is a strongly simplified training loop
    for epoch in range(10):
        torch.manual_seed(epoch)
        train_loss_epoch = 0
        train_accuracy_epoch = torch.zeros(10)
        for iteration in range(20):
            train_loss = 0.9 ** (epoch + iteration / 20.)
            train_loss_epoch += train_loss
            train_logger.add_scalar('loss', train_loss, global_step=global_step)
            train_accuracy = epoch / 10. + torch.randn(10)
            train_accuracy_epoch += train_accuracy
            global_step += 1

        print('loss=%0.3f, epoch=%d' % (train_loss_epoch, epoch))

        # Log the training accuracy
        train_accuracy_epoch /= 20  # Average accuracy over all iterations
        train_logger.add_scalar('accuracy', train_accuracy_epoch.mean(), global_step=global_step)
        print('accuracy=%0.3f, epoch=%d' % (train_accuracy_epoch.mean(), epoch))

        torch.manual_seed(epoch)
        valid_accuracy_epoch = torch.zeros(10)
        for iteration in range(10):
            validation_accuracy = epoch / 10. + torch.randn(10)
            valid_accuracy_epoch += validation_accuracy

        # Log the validation accuracy
        valid_accuracy_epoch /= 10  # Average accuracy over all iterations
        valid_logger.add_scalar('accuracy', valid_accuracy_epoch.mean(), global_step=global_step)
        print('accuracy=%0.3f, epoch=%d' % (valid_accuracy_epoch.mean(), epoch))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
