import torch
import torch.nn as nn
import torch.optim as optim
from os import path
import torch.utils.tensorboard as tb
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from state_agent.state_model import IceHockeyKartNetWithActions, load_model, save_model
from state_agent.state_utils import load_data_from_multiple_pickles_in_batches_v2_norm


def train(args):
    model = IceHockeyKartNetWithActions()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.continue_training = True
    if args.continue_training:
        model = load_model()
    model.to(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    num_epochs = 50
    best_validation_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 10

    criterion_regression = nn.MSELoss()  # For acceleration and steer

    for epoch in range(num_epochs):
        train_data_loader = load_data_from_multiple_pickles_in_batches_v2_norm("data/train", 512)
        valid_data_loader = load_data_from_multiple_pickles_in_batches_v2_norm("data/valid", 512)

        model.train()
        total_loss = 0.0
        total_loss_acceleration = 0.0
        total_loss_steer = 0.0
        total_loss_brake = 0.0
        total_samples = 0

        # Training loop
        for item in train_data_loader:
            player_game_features = item[0]
            actions = item[1]
            player_game_features = player_game_features.to(device)
            target_accelerations = actions[:, 0].unsqueeze(1)
            target_steers = actions[:, 1].unsqueeze(1)
            target_brakes = actions[:, 2].unsqueeze(1)
            target_accelerations = target_accelerations.to(device)
            target_steers = target_steers.to(device)
            target_brakes = target_brakes.to(device)

            optimizer.zero_grad()
            # Forward pass
            acceleration_preds, steer_preds, brake_preds = model(player_game_features)

            # Compute loss
            loss_acceleration = criterion_regression(acceleration_preds, target_accelerations)
            loss_steer = criterion_regression(steer_preds, target_steers)
            loss_brake = criterion_regression(brake_preds, target_brakes.float())
            loss = loss_acceleration + loss_steer + loss_brake

            # Accumulate individual losses
            total_loss_acceleration += loss_acceleration.item()
            total_loss_steer += loss_steer.item()
            total_loss_brake += loss_brake.item()
            total_loss += loss.item()
            total_samples += player_game_features.size(0)

            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Logging training loss
        average_loss = total_loss / total_samples
        average_loss_acceleration = total_loss_acceleration / total_samples
        average_loss_steer = total_loss_steer / total_samples
        average_loss_brake = total_loss_brake / total_samples
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {average_loss:.6f}')

        if train_logger is not None:
            train_logger.add_scalar('loss', average_loss, global_step=epoch)
            train_logger.add_scalar('loss_acceleration', average_loss_acceleration, global_step=epoch)
            train_logger.add_scalar('loss_steer', average_loss_steer, global_step=epoch)
            train_logger.add_scalar('loss_brake', average_loss_brake, global_step=epoch)

        # Validation loop
        model.eval()
        total_validation_loss = 0.0
        total_validation_loss_acceleration = 0.0
        total_validation_loss_steer = 0.0
        total_validation_loss_brake = 0.0
        total_validation_samples = 0
        with torch.no_grad():
            for item in valid_data_loader:
                player_game_features = item[0]
                actions = item[1]
                player_game_features = player_game_features.to(device)
                target_accelerations = actions[:, 0].unsqueeze(1)
                target_steers = actions[:, 1].unsqueeze(1)
                target_brakes = actions[:, 2].unsqueeze(1)
                target_accelerations = target_accelerations.to(device)
                target_steers = target_steers.to(device)
                target_brakes = target_brakes.to(device)

                # Forward pass
                acceleration_preds, steer_preds, brake_preds = model(player_game_features)

                # Compute loss
                loss_acceleration = criterion_regression(acceleration_preds, target_accelerations)
                loss_steer = criterion_regression(steer_preds, target_steers)
                loss_brake = criterion_regression(brake_preds, target_brakes.float())
                loss = loss_acceleration + loss_steer + loss_brake

                # Accumulate individual validation losses
                total_validation_loss_acceleration += loss_acceleration.item()
                total_validation_loss_steer += loss_steer.item()
                total_validation_loss_brake += loss_brake.item()
                total_validation_loss += loss.item()
                total_validation_samples += player_game_features.size(0)

            # Average validation losses
            average_validation_loss = total_validation_loss / total_validation_samples
            average_validation_loss_acceleration = total_validation_loss_acceleration / total_validation_samples
            average_validation_loss_steer = total_validation_loss_steer / total_validation_samples
            average_validation_loss_brake = total_validation_loss_brake / total_validation_samples

            # Print validation loss
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {average_validation_loss:.6f}')

            if valid_logger is not None:
                valid_logger.add_scalar('loss', average_validation_loss, global_step=epoch)
                valid_logger.add_scalar('loss_acceleration', average_validation_loss_acceleration,
                                        global_step=epoch)
                valid_logger.add_scalar('loss_steer', average_validation_loss_steer, global_step=epoch)
                valid_logger.add_scalar('loss_brake', average_validation_loss_brake, global_step=epoch)

        scheduler.step(average_validation_loss)

        # Early stopping check
        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            epochs_no_improve = 0
            # Save the best model
            save_model(model)
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epochs.")

        if epochs_no_improve >= early_stopping_patience:
            print(f"Stopping early after {epochs_no_improve} epochs without improvement.")
            break

    # Save the trained model
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')

    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)


