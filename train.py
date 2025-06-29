import gymnasium as gym
import torch
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

from trajectory_env import DroneTrajectoryEnv
from ppo_agent import PPOAgent
from visualizer import DroneAnimation

# --- Hyperparameters ---
config = {
    "learning_rate": 3e-4,
    "num_envs": 16,
    "num_steps": 2048,
    "total_timesteps": 5_000_000,
    "update_epochs": 10,
    "minibatch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "eval_every_n_episodes": 16*10
}
config["batch_size"] = int(config["num_envs"] * config["num_steps"])

def make_env(seed):
    def thunk():
        env = DroneTrajectoryEnv()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env
    return thunk

def evaluate_and_visualize(agent, device):
    print("\n--- Değerlendirme ve Görselleştirme Başladı ---")
    eval_env = DroneTrajectoryEnv()
    
    drone_history = {'pos': [], 'rotors': []}
    ref_history = {'pos': []}
    
    arm_len = eval_env.drone.ARM_LEN
    rotor_pos_body = np.array([
        [arm_len, 0, 0], [-arm_len, 0, 0], [0, arm_len, 0], [0, -arm_len, 0]
    ])

    obs, _ = eval_env.reset()
    done = False
    while not done:
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device).unsqueeze(0))
        
        obs, _, terminated, truncated, _ = eval_env.step(action.cpu().numpy()[0])
        done = terminated or truncated

        drone_history['pos'].append(eval_env.drone.pos.copy())
        drone_history['rotors'].append(eval_env.drone.quat.apply(rotor_pos_body) + eval_env.drone.pos)
        ref_pos, _ = eval_env.trajectory.get_state(eval_env.time)
        ref_history['pos'].append(ref_pos)
    
    anim = DroneAnimation(drone_history, ref_history)
    anim.run()
    print("--- Değerlendirme Tamamlandı ---\n")


if __name__ == "__main__":
    run_name = f"drone_ppo_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    envs = gym.vector.AsyncVectorEnv(
        [make_env(i) for i in range(config['num_envs'])]
    )
    
    agent = PPOAgent(envs, device, **config)
    
    # train loop
    global_step = 0
    start_time = time.time()
    
    obs, _ = envs.reset()
    next_obs = torch.Tensor(obs).to(device)
    next_done = torch.zeros(config['num_envs']).to(device)
    
    num_updates = config['total_timesteps'] // config['batch_size']
    
    for update in range(1, num_updates + 1):
        # Ortamdan veri topla
        for step in range(0, config['num_steps']):
            global_step += 1 * config['num_envs']
            agent.obs[step] = next_obs
            agent.dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.agent.get_action_and_value(next_obs)
                agent.values[step] = value.flatten()
            agent.actions[step] = action
            agent.logprobs[step] = logprob

            obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            agent.rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(obs).to(device), torch.Tensor(done).to(device)

            if "final_info" in info:
                for item in info["final_info"]:
                    if item and "episode" in item:
                        print(f"global_step={global_step}, episode_reward={item['episode']['r']:.2f}")
                        writer.add_scalar("charts/episode_reward", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episode_length", item["episode"]["l"], global_step)
                        writer.add_scalar("charts/position_error", item["pos_error"], global_step)
                        break

        # agent update
        v_loss, pg_loss, ent_loss = agent.update()
        
        # Metrikleri logla
        writer.add_scalar("losses/value_loss", v_loss, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss, global_step)
        writer.add_scalar("losses/entropy", ent_loss, global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        print(f"SPS: {sps}")

        if update % (config['eval_every_n_episodes'] // config['num_envs']) == 0:
             evaluate_and_visualize(agent.agent, device)

    envs.close()
    writer.close()
