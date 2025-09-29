import gymnasium_robotics
import gymnasium as gym
import numpy as np
import torch
from torchvision.io import write_video
import os
import minari

env = gym.make('AntMaze_Medium-v4')
env.reset()
env.unwrapped.ant_env.render_mode = 'rgb_array'
# D4RL/antmaze/medium-play-v1
# D4RL/antmaze/umaze-diverse-v1
# D4RL/antmaze/large-diverse-v1
# D4RL/antmaze/large-play-v1
# D4RL/antmaze/medium-diverse-v1
# D4RL/antmaze/umaze-v1
dataset = minari.load_dataset("D4RL/antmaze/medium-play-v1")
# observations
#     observation (1001, 27)
#     desired_goal (1001, 2)
#     achieved_goal (1001, 2)
# actions (1000, 8)
# infos
#     goal (1001, 2)
#     qpos (1001, 15)
#     qvel (1001, 14)

states2compare = {}
num_video = 1
save_folder = "./data_gen_antmaze_videos"
os.makedirs(save_folder, exist_ok=True)

start_idx = 0
for i, episode_data in enumerate(dataset.iterate_episodes()):
    env.reset()
    traj_acts = episode_data.actions
    qpos = episode_data.infos['qpos']
    qvel = episode_data.infos['qvel']
    frames = []
    for t in range(1, len(qpos)):
        qpos_t = qpos[t]
        qvel_t = qvel[t]
        env.unwrapped.ant_env.set_state(qpos_t, qvel_t)
        obs = env.render().copy()
        frames.append(torch.from_numpy(obs))
    video_tensor = torch.stack(frames)
    write_video(f'{save_folder}/output_state_video_{i}.mp4', video_tensor, fps=30) # TODO: delete this after confirm
    
    states2compare[i] = {
        "states": np.concatenate([qpos[1:], qvel[1:]], axis=1), # remove the first state to align with actions
    }
    if i >= num_video:
        break

env.close()

env = gym.make('AntMaze_Medium-v4')
env.unwrapped.ant_env.render_mode = 'rgb_array'
# env.unwrapped.ant_env._include_cfrc_ext_in_observation = False
env.reset()
dataset = minari.load_dataset("D4RL/antmaze/medium-play-v1")

for i, episode_data in enumerate(dataset.iterate_episodes()):
    env.reset()
    traj_acts = episode_data.actions
    qpos = episode_data.infos['qpos']
    qvel = episode_data.infos['qvel']
    frames = []

    qpos_0 = qpos[0]
    qvel_0 = qvel[0]
    env.unwrapped.ant_env.set_state(qpos_0, qvel_0)

    states = []
    for t in range(len(traj_acts)):
        action = traj_acts[t]
        ob, _, _, _, _ = env.step(action)
        # state = ob
        state = np.concatenate([ob["achieved_goal"], ob["observation"]], axis=0)
        states.append(state)
        obs = env.render().copy()
        frames.append(torch.from_numpy(obs))
    video_tensor = torch.stack(frames)
    write_video(f'{save_folder}/output_action_video_{i}.mp4', video_tensor, fps=30) # TODO: delete this after confirm
    
    states = np.stack(states)
    states2compare[i].update({
        "actions": states,
    })
    if i >= num_video:
        break

env.close()

import matplotlib.pyplot as plt
for i, d in states2compare.items():
    states = d.get("states")
    actions = d.get("actions")
    diffs = states - actions
    norms = np.linalg.norm(diffs, axis=1)

    fig = plt.figure(figsize=(8, 3))
    plt.plot(norms)
    plt.title(f"AntMaze state diff L2 norms (first 29 dims) — traj {i}")
    plt.xlabel("Frame")
    plt.ylabel("||Δstate||")
    plt.tight_layout()
    out_path = os.path.join(save_folder, f"norms_traj_{i}.png")
    plt.savefig(out_path, dpi=150)

    plt.close(fig)