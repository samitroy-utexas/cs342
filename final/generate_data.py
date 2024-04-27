import itertools
import os
import subprocess

# Define the ranges for each variable
categories = ['train', 'valid']
#frame_options = range(1200, 20001, 1800)  # Example range, adjust as needed
max_score_options = range(3, 101, 10)  # Example range, adjust as needed
count_options = range(11, 100)  # Example range, adjust as needed
agents = [
    'geoffrey_agent',
    'yann_agent',
    'yoshua_agent',
    'AI'
]

# Create the data directories if they don't exist
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/valid', exist_ok=True)

# Generate the commands, ensuring agent1 and agent2 are not the same
count = 515
frame_options = 1000
while count < 100000:
    for category, agent2 in itertools.product(categories, agents):
        filename = f'jurgen_{agent2}_{category}_{count}.pkl'
        directory = f'data\\{category}'
        filepath = os.path.join(directory, filename)
        command = [
            'python', '-m', 'tournament.runner',
            '-s', filepath,
            '-f', str(frame_options),
            '-p', '2',
            '-m', str(100),
            '-j', '100',
            'jurgen_agent', agent2
        ]

        # Print the command to be executed (for debugging purposes)
        print(' '.join(command))

        # Execute the command
        subprocess.run(command, shell=True)
        count = count + 1
        frame_options = frame_options + 10

