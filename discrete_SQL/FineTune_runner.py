import subprocess
import random
import time
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
wandb.login(key="a2ff527595e001c6604deef5f2f3a8ed97c08407")

set_up = {
    "--env-id": "Walker2d-v4",
    "--wandb-entity": "angela-h",
    "--wandb-project-name": "SQL_hparam_finetune",
    "--track": True,
    "--total-timesteps": "100000",
    # "--autotune": True,
    "--learning-starts": "500",
    "--seed": None,  # Placeholder
}

seeds = [42, 128, 456, 864, 10248]

hyperparameter_space = {
    "--q-lr": [1e-3, 1e-4],  # Learning rate for the Q network
    #"--batch-size": [64, 128],  # Batch size for training
    #"--gamma": [0.99, 0.95, 0.90],  # Discount factor for future rewards
    #"--tau": [0.005, 0.01],  # Target network update rate
    #"--alpha": [0.2, 0.3],  # Entropy regularization coefficient
}

def sample_hparam(hyperparameter_space):
    """Randomly sample a set of hparam from the defined space."""
    return {param: random.choice(values) for param, values in hyperparameter_space.items()}

def run(set_up, hparams,seed):
    """Run training script with specified hparam."""
    hparams_str = "_".join([f"{k}{v}" for k, v in hparams.items()])
    run_name = f"_{hparams_str}"

    command = ["python", "discrete_SQL.py", "--exp-name", run_name]
    for param, value in set_up.items():
        if param != '--seed':
            command.extend([param, str(value)])
    command.extend(["--seed", str(seed)])

    for param, value in hparams.items():
        command.extend([param, str(value)])

    print(f"Executing command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    episodic_returns = [float(line.split('=')[-1].strip(']').strip('[')) for line in result.stdout.split('\n') if "episodic_return" in line]
    final_episodic_return = episodic_returns[-1] if episodic_returns else 0
    print(f"Final episodic return: {final_episodic_return}")

    return final_episodic_return

def hparam_tuning(set_up, hparam_space, seeds, num_trials=3):
    results = []
    for trial in range(num_trials):
        hparams = sample_hparam(hparam_space)
        final_returns = []
        for seed in seeds:
          start = time.time()
          print(f"Trial {trial+1}, Seed {seed}: Running training with hparam: {hparams}")
          final_return = run(set_up, hparams, seed)
          end = time.time()
          print(f"Trial {trial+1} Time used:", end - start, "seconds")
          final_returns.append(final_return)

        avg_return = np.mean(final_returns)
        std_return = np.std(final_returns)
        results.append({"hparams": hparams, "avg_return": avg_return, "std_return": std_return})
        print(f"Hparam: {hparams}, Avg Return: {avg_return}, Std Return: {std_return}")
    return results

def plot_results(results):
    sns.set_theme(style="whitegrid")
    avg_returns = [result["avg_return"] for result in results]
    std_returns = [result["std_return"] for result in results]
    hparams_labels = [str(result["hparams"]) for result in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=hparams_labels, y=avg_returns, yerr=std_returns, capsize=.2, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title('Hyperparameter Tuning Results')
    ax.set_ylabel('Average Final Return')
    ax.set_xlabel('Hyperparameters')
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    results = hparam_tuning(set_up, hyperparameter_space,seeds)