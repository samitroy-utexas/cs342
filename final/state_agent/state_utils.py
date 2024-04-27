import pickle
import torch
import numpy as np
import os

# Move computation to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_large_pickle_file(file_path, chunk_size=8192):
    with open(file_path, 'rb') as f:
        while True:
            chunk = []
            try:
                # Load a chunk of data from the pickle file
                for _ in range(chunk_size):
                    item = pickle.load(f)
                    chunk.append(item)
            except EOFError:
                # If EOFError is encountered, it means we've reached the end of the file
                pass

            if not chunk:
                # If the chunk is empty, it means there's no more data to read, so break
                break
            else:
                # Yield the current chunk, which may be less than chunk_size if at the end of file
                yield chunk

def load_data_from_pickle(filename):
    generator = read_large_pickle_file(filename)
    data = []
    for chunk in generator:
        data.extend(chunk)
    return data

def limit_period(val):
    # Function to limit the angle difference to the range [-1, 1]
    val = val % (2 * np.pi)
    if val > np.pi:
        val -= 2 * np.pi
    return val / np.pi


def extract_features_and_actions(team_id, pstate, opponent_state, soccer_state):
    # features of ego-vehicle
    kart_front = torch.tensor(pstate['kart']['front'], dtype=torch.float32)[[0, 2]]
    kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
    kart_direction = (kart_front - kart_center) / torch.norm(kart_front - kart_center)
    kart_angle = torch.atan2(kart_direction[1], kart_direction[0])
    kart_velocity = torch.tensor(pstate['kart']['velocity'], dtype=torch.float32)[[0, 2]]
    kart_speed = torch.norm(kart_velocity)

    # features of soccer
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center - kart_center)
    kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0])
    kart_to_puck_angle_difference = limit_period(kart_angle - kart_to_puck_angle)

    # features of the goal line
    goal_line_center = torch.tensor(soccer_state['goal_line'][team_id], dtype=torch.float32)[:, [0, 2]].mean(dim=0)
    puck_to_goal_line = (goal_line_center - puck_center) / torch.norm(goal_line_center - puck_center)
    puck_to_goal_line_angle = torch.atan2(puck_to_goal_line[1], puck_to_goal_line[0])
    kart_to_goal_line_angle_difference = limit_period(kart_angle - puck_to_goal_line_angle)

    # Angle from puck to goal line
    puck_to_goal_angle = torch.atan2(goal_line_center[1] - puck_center[1], goal_line_center[0] - puck_center[0])

    # Distance from the kart to the puck
    kart_to_puck_distance = torch.norm(puck_center - kart_center)

    # Additional features
    kart_to_goal_direction = (goal_line_center - kart_center) / torch.norm(goal_line_center - kart_center)
    kart_to_goal_angle = torch.atan2(kart_to_goal_direction[1], kart_to_goal_direction[0])
    kart_to_goal_angle_difference = limit_period(kart_angle - kart_to_goal_angle)

    # Combine all features into a single tensor
    features = torch.cat([
        kart_center,  # Kart's location
        kart_direction,  # Kart's direction
        torch.tensor([kart_angle]),  # Kart's angle
        kart_velocity,  # Kart's velocity
        torch.tensor([kart_speed]),  # Kart's speed
        puck_center,  # Puck's location
        torch.tensor([kart_to_puck_angle,  # Angle to puck from kart
                      kart_to_puck_angle_difference,
                      # Angle difference between kart's orientation and direction to puck
                      puck_to_goal_line_angle,  # Angle from puck to goal line
                      kart_to_goal_line_angle_difference,
                      # Angle difference between kart's orientation and direction to goal line
                      puck_to_goal_angle,  # Angle from puck to goal
                      kart_to_goal_angle_difference,  # Angle difference between kart's orientation and goal line
                      kart_to_puck_distance  # Distance from kart to puck
                      ])
    ])

    return features


def extract_actions_from_state(team_id, actions):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Default values for each action
    default_value = torch.tensor(0.0, dtype=torch.float32, device=device)

    # Extract actions if available, otherwise default to zeros
    if actions:
        # Extract actions for the specified team_id, if they exist
        if actions and len(actions) > team_id and 'acceleration' in actions[team_id]:
            acceleration = actions[team_id]['acceleration'].clone().detach().to(device)
        else:
            acceleration = default_value

        if actions and len(actions) > team_id and 'steer' in actions[team_id]:
            steer = actions[team_id]['steer'].clone().detach().to(device)
        else:
            steer = default_value

        if actions and len(actions) > team_id and 'brake' in actions[team_id]:
            brake = actions[team_id]['brake'].clone().detach().to(device)
        else:
            brake = default_value
    else:
        acceleration = torch.tensor(0.0, dtype=torch.float32, device=device)
        steer = torch.tensor(0.0, dtype=torch.float32, device=device)
        brake = torch.tensor(0.0, dtype=torch.float32, device=device)

    # Combine actions into a single tensor
    acceleration = torch.nan_to_num(acceleration, nan=0.0)
    steer = torch.nan_to_num(steer, nan=0.0)
    brake = torch.nan_to_num(brake, nan=0.0)
    actions = torch.tensor([acceleration, steer, brake], dtype=torch.float32)
    return actions


def normalize_data(data):
    # Calculate mean and standard deviation for each feature
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # Avoid division by zero by adding a small epsilon value
    epsilon = 1e-8
    std += epsilon

    # Normalize data using z-score normalization
    normalized_data = (data - mean) / std
    return normalized_data


def numpy_array_to_tensor_list(numpy_array):
    tensor_list = [torch.tensor(item, dtype=torch.float32) for item in numpy_array]
    return tensor_list


def load_data_from_multiple_pickles_in_batches_v2_norm(pickle_dir, batch_size):
    batch_features = []
    batch_actions = []

    # Iterate through pickle files in the directory
    count = 0
    for pickle_file in os.listdir(pickle_dir):
        count = count + 1
        #print(f"======================================={count}===={pickle_file}=======================================")
        if pickle_file.endswith('.pkl'):
            datas = load_data_from_pickle(os.path.join(pickle_dir, pickle_file))
            for data in datas:
                for player in data["team1_state"]:
                    features = extract_features_and_actions(0, player, data["team2_state"], data["soccer_state"])
                    actions = extract_actions_from_state(player['kart']['player_id'], data['actions'])

                    if (not torch.isnan(actions).any() and not torch.isnan(features).any()
                            and not torch.isinf(actions).any() and not torch.isinf(features).any()):
                        # Move features and actions to GPU
                        features = features.to(device)
                        actions = actions.to(device)

                        # Append features and actions to batches
                        batch_features.append(features)
                        batch_actions.append(actions)

                    # When the batch is full, yield it and reset the lists
                    if len(batch_features) == batch_size:
                        batch_indices = list(range(len(batch_features)))
                        batch_features = [batch_features[i].cpu() for i in batch_indices]
                        batch_actions = [batch_actions[i].cpu() for i in batch_indices]

                        # Convert lists to numpy arrays
                        x = np.array(batch_features)
                        y = batch_actions
                        y = torch.stack(y, dim=0)

                        # Normalize features
                        x_normalized = normalize_data(x)

                        # Convert normalized features to list of tensors
                        x_normalized_tensor_list = numpy_array_to_tensor_list(x_normalized)
                        x_normalized_tensor_list = torch.stack(x_normalized_tensor_list, dim=0)

                        # Move data to GPU
                        x_normalized_tensor_list = x_normalized_tensor_list.to(device)
                        y = y.to(device)

                        yield x_normalized_tensor_list, y
                        batch_features = []
                        batch_actions = []

            # Yield any remaining data as the last batch (if not empty)
            if batch_features and batch_actions:
                # Shuffle the batches
                batch_indices = list(range(len(batch_features)))
                #random.shuffle(batch_indices)
                batch_features = [batch_features[i].cpu() for i in batch_indices]
                batch_actions = [batch_actions[i].cpu() for i in batch_indices]

                # Convert lists to numpy arrays
                x = np.array(batch_features)
                y = batch_actions
                y = torch.stack(y, dim=0)

                # Normalize features
                x_normalized = normalize_data(x)

                # Convert normalized features to list of tensors
                x_normalized_tensor_list = numpy_array_to_tensor_list(x_normalized)
                x_normalized_tensor_list = torch.stack(x_normalized_tensor_list, dim=0)

                # Move data to GPU
                x_normalized_tensor_list = x_normalized_tensor_list.to(device)
                y = y.to(device)

                yield x_normalized_tensor_list, y

