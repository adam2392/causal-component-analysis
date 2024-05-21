import os

n_repeats = 50  # number of different training seeds to use from 1-100
sim_names = ["chain_graph", 
             'collider_graph', 
            #  'nonmarkov_graph'
             ]
sim_graphs = {
    'chain_graph': [
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ],
    'collider_graph': [
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0],
    ],
    'nonmarkov_graph': [
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ],
}
intervention_targets = {
    'chain_graph': [
        [
            [0, 0, 0], 
            [0, 0, 1],
            [0, 0, 1],
        ],
    ],
    'collider_graph': [
        [
            [0, 0, 0],
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
        ],
    ],
    'nonmarkov_graph': [
        [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
        ],
    ]
}
noise_shift_type = 'mean-std'
batch_size = 4096
max_epochs = 150
n_samples = [200_000, 500_000]

curr_dir = os.getcwd()

with open(f"{curr_dir}/parameters_cdrl.txt", "w") as file:
    for sim_name in sim_names:
        for training_seed in range(1, n_repeats + 1):
            current_sim_graph = sim_graphs[sim_name]
            current_intervention_targets = intervention_targets[sim_name]
            line = f'--sim-name={sim_name} --training-seed={training_seed} --sim_graph={current_sim_graph} --num-samples={n_samples} --intervention_targets={current_intervention_targets} --noise_shift_type={noise_shift_type} --batch_size={batch_size} --max_epochs={max_epochs}\n'
            file.write(line)

print("Parameters file generated successfully.")
