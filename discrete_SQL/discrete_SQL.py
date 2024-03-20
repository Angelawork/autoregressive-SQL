# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import QTransformer, ARQ
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="advantage-diffusion",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default='swish',
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Hopper-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--qnet", type=str, default="mlp", help="the architecture of the Q network ('mlp' or 'transformer')")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning")
    parser.add_argument("--q-lr", type=float, default=1e-4,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--q-steps", type=int, default=1,
        help="Number of Q steps per env step")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--alpha", type=float, default=0.05,
            help="Entropy regularization coefficient.")
    parser.add_argument("--exploration-alpha", type=float, default=0.05,
            help="Entropy regularization coefficient.")
    parser.add_argument("--separate-explore-alpha", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use different alpha for exploration")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--target-entropy", type=float, default=2.0)
    parser.add_argument("--a_bins", type=int, default=20)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--hdim", type=int, default=512)
    args = parser.parse_args()
    # fmt: on
    return args

def make_env(env_id, seed, idx, capture_video, run_name,step_list=[100000,150000,175000,200000,250000]):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}",step_trigger= lambda x: x in step_list)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    action_min = float(envs.single_action_space.low[0])
    action_max = float(envs.single_action_space.high[0])
    s_dim = envs.single_observation_space.shape[0]
    a_dim = envs.single_action_space.shape[0]
    
    if args.qnet == 'transformer':
        qf1 = QTransformer(s_dim=s_dim, a_dim=a_dim, a_bins=args.a_bins, alpha=args.alpha, num_layers=args.num_layers, nhead=args.nhead, action_min=action_min, action_max=action_max, hdim=args.hdim).to(device)
        qf2 = QTransformer(s_dim=s_dim, a_dim=a_dim, a_bins=args.a_bins, alpha=args.alpha, num_layers=args.num_layers, nhead=args.nhead, action_min=action_min, action_max=action_max, hdim=args.hdim).to(device)
        qf1_target = QTransformer(s_dim=s_dim, a_dim=a_dim, a_bins=args.a_bins, alpha=args.alpha, num_layers=args.num_layers, nhead=args.nhead, action_min=action_min, action_max=action_max, hdim=args.hdim).to(device)
        qf2_target = QTransformer(s_dim=s_dim, a_dim=a_dim, a_bins=args.a_bins, alpha=args.alpha, num_layers=args.num_layers, nhead=args.nhead, action_min=action_min, action_max=action_max, hdim=args.hdim).to(device)
    else:
        qf1 = ARQ(s_dim=s_dim, a_dim=a_dim, a_bins=args.a_bins, alpha=args.alpha, action_min=action_min, action_max=action_max).to(device)
        qf2 = ARQ(s_dim=s_dim, a_dim=a_dim, a_bins=args.a_bins, alpha=args.alpha, action_min=action_min, action_max=action_max).to(device)
        qf1_target = ARQ(s_dim=s_dim, a_dim=a_dim, a_bins=args.a_bins, alpha=args.alpha, action_min=action_min, action_max=action_max).to(device)
        qf2_target = ARQ(s_dim=s_dim, a_dim=a_dim, a_bins=args.a_bins, alpha=args.alpha, action_min=action_min, action_max=action_max).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    
    if args.autotune:
        target_entropy = args.target_entropy#-torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
        qf1.alpha = alpha
        qf2.alpha = alpha
        qf1_target.alpha = alpha
        qf2_target.alpha = alpha
    else:
        alpha = args.alpha
    if not args.separate_explore_alpha:
        args.exploration_alpha = alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    entropy = 0.0
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            actions = torch.Tensor(actions).to(device)
            actions = qf1.dequantize_action(qf1.quantize_action(actions)).detach().cpu().numpy()
        else:
            actions, log_prob = qf1.sample_action(torch.Tensor(obs).to(device), return_entropy=True, exploration_alpha=args.exploration_alpha)
            entropy = entropy + log_prob.mean().item()
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs
        
        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            for steps in range(args.q_steps):
                data = rb.sample(args.batch_size)
                if args.qnet == 'mlp':
                    with torch.no_grad():
                        Q_next_f1, V_next_f1, _ = qf1_target.forward_once(data.next_observations)
                        Q_next_f2, V_next_f2, _ = qf2_target.forward_once(data.next_observations)
                        # Min for all Q
                        Q_next = torch.min(Q_next_f1, Q_next_f2)
                        V_next_f1 = torch.logsumexp(Q_next / alpha, dim=1, keepdim=True) * alpha
                        V_next_f2 = torch.logsumexp(Q_next / alpha, dim=1, keepdim=True) * alpha
                        
                        Q1, V_f1, _, Q1_final = qf1_target(data.observations, data.actions)
                        Q2, V_f2, _, Q2_final = qf2_target(data.observations, data.actions)
                        # Min for all Q
                        Q = torch.min(Q1, Q2)
                        V_f1 = torch.logsumexp(Q / alpha, dim=2) * alpha
                        V_f2 = torch.logsumexp(Q / alpha, dim=2) * alpha
                        
                        V_next_f1 = data.rewards + (1 - data.dones) * args.gamma * V_next_f1
                        V_next_f2 = data.rewards + (1 - data.dones) * args.gamma * V_next_f2
                        V_f1 = torch.cat([V_f1, V_next_f1[:,:1]], dim=1)[:,1:]
                        V_f2 = torch.cat([V_f2, V_next_f2[:,:1]], dim=1)[:,1:]
                        td_target = torch.min(V_f1, V_f2)
                        action_idx = qf1.quantize_action(data.actions)
                    Q1, _, _, Q1_final = qf1(data.observations, data.actions)
                    Q2, _, _, Q2_final = qf2(data.observations, data.actions)
                    qf1_a_values = torch.gather(Q1, 2, action_idx.unsqueeze(-1))
                    qf2_a_values = torch.gather(Q2, 2, action_idx.unsqueeze(-1))
                    #qf1_final_loss = F.mse_loss(Q1_final, V_next_f1.detach())
                    #qf2_final_loss = F.mse_loss(Q2_final, V_next_f2.detach())
                    qf1_loss = F.mse_loss(qf1_a_values.squeeze(2), td_target.detach())
                    qf2_loss = F.mse_loss(qf2_a_values.squeeze(2), td_target.detach())
                    qf_loss = qf1_loss + qf2_loss #+ qf1_final_loss + qf2_final_loss
                    
                elif args.qnet == 'transformer':
                    with torch.no_grad():
                        _, V_next_f1, _ = qf1_target(data.next_observations.unsqueeze(1))
                        _, V_next_f2, _ = qf2_target(data.next_observations.unsqueeze(1))
                        Q1, V_f1, _ = qf1_target(data.observations.unsqueeze(1), data.actions[:,:-1].unsqueeze(2))
                        Q2, V_f2, _ = qf2_target(data.observations.unsqueeze(1), data.actions[:,:-1].unsqueeze(2))
                        V_f1[:,:-1] = V_f1[:,1:]
                        V_f1[:,-1] = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * V_next_f1[:,0]
                        V_f2[:,:-1] = V_f2[:,1:]
                        V_f2[:,-1] = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * V_next_f2[:,0]
                        td_target = torch.min(V_f1, V_f2)
                        action_idx = qf1.quantize_action(data.actions)
                    Q1, _, _ = qf1(data.observations.unsqueeze(1), data.actions[:,:-1].unsqueeze(2))
                    Q2, _, _ = qf2(data.observations.unsqueeze(1), data.actions[:,:-1].unsqueeze(2))
                    qf1_a_values = torch.gather(Q1, 2, action_idx.unsqueeze(-1))
                    qf2_a_values = torch.gather(Q2, 2, action_idx.unsqueeze(-1))
                    qf1_loss = F.mse_loss(qf1_a_values.squeeze(2), td_target.detach())
                    qf2_loss = F.mse_loss(qf2_a_values.squeeze(2), td_target.detach())
                    qf_loss = qf1_loss + qf2_loss
                
                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()
                
            if args.autotune:
                with torch.no_grad():
                    _, log_prob = qf1.sample_action(data.observations, return_entropy=True, exploration_alpha=alpha)
                alpha_loss = (-log_alpha.exp() * (-log_prob + target_entropy)).mean()

                a_optimizer.zero_grad()
                alpha_loss.backward()
                a_optimizer.step()
                alpha = log_alpha.exp().item()
                args.exploration_alpha = alpha
                qf1.alpha = alpha
                qf2.alpha = alpha
                qf1_target.alpha = alpha
                qf2_target.alpha = alpha
            
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                
            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/TD_target", td_target.mean().item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("losses/entropy", entropy / 100, global_step)
                entropy = 0.0
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
