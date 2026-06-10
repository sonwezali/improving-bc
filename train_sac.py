import os
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import random
from tqdm import tqdm

import lidbox_env
from models import SACGaussianPolicy, SACCritic, BCPolicy
from eval import evaluate_policy


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)),
                torch.FloatTensor(np.array(actions)),
                torch.FloatTensor(np.array(rewards)).unsqueeze(1),
                torch.FloatTensor(np.array(next_states)),
                torch.FloatTensor(np.array(dones)).unsqueeze(1))

    def __len__(self):
        return len(self.buffer)


def compute_actor_loss(actor, critic1, critic2, bc_model, states, alpha, bc_lambda_now):
    actions_pi, log_probs_pi, mean_pi, _ = actor.sample(states)
    q1_pi = critic1(states, actions_pi)
    q2_pi = critic2(states, actions_pi)
    q_pi = torch.min(q1_pi, q2_pi)
    q_scale = q_pi.abs().mean().detach().clamp(min=1e-3)
    sac_actor_loss = (alpha * log_probs_pi - q_pi / q_scale).mean()

    with torch.no_grad():
        bc_target = bc_model(states)
    det_action = torch.tanh(mean_pi)
    bc_loss = F.mse_loss(det_action, bc_target)

    return sac_actor_loss + bc_lambda_now * bc_loss, bc_loss, q_scale


def train_sac(bc_ckpt_path="bc_policy.pt", num_episodes=750, batch_size=1024,
              gamma=0.99, tau=0.005, alpha=0.05, lr=0.0003, bc_lambda=10.0,
              bc_lambda_final=5.0, bc_lambda_anneal_episodes=200,
              actor_freeze_episodes=10,
              online_mix_target=0.75, online_mix_ramp_episodes=100,
              demos_path="demos.npy", save_path="sac_policy.pt",
              ckpt_dir="checkpoints", ckpt_every=50,
              eval_every=25, eval_episodes=20, eval_seed=591,
              warmup_steps=2000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(ckpt_dir, exist_ok=True)

    env = lidbox_env.LidBoxEnv(render_mode="offscreen")

    bc_ckpt = torch.load(bc_ckpt_path, map_location=device, weights_only=False)
    state_mean = torch.from_numpy(bc_ckpt["state_mean"]).float().to(device)
    state_std = torch.from_numpy(bc_ckpt["state_std"]).float().to(device)

    actor = SACGaussianPolicy().to(device)
    bc_model = BCPolicy().to(device)
    bc_model.load_state_dict(bc_ckpt["model_state_dict"])
    bc_model.eval()  

    actor_state = actor.state_dict()
    bc_state = bc_model.state_dict()
    for key in actor_state:
        if key in bc_state and actor_state[key].shape == bc_state[key].shape:
            actor_state[key] = bc_state[key]
    last = max(int(k.split(".")[1]) for k in bc_state if k.startswith("net.") and k.endswith(".weight"))
    if actor_state["mean_head.weight"].shape == bc_state[f"net.{last}.weight"].shape:
        actor_state["mean_head.weight"] = bc_state[f"net.{last}.weight"]
        actor_state["mean_head.bias"] = bc_state[f"net.{last}.bias"]
    else:
        print("[warn] BC final-layer shape mismatch; mean_head left at init "
              "(warm-start will NOT reproduce BC).")
    actor.load_state_dict(actor_state)

    torch.nn.init.zeros_(actor.log_std_head.weight)
    torch.nn.init.constant_(actor.log_std_head.bias, -3.0)

    critic1 = SACCritic().to(device)
    critic2 = SACCritic().to(device)
    critic1_target = SACCritic().to(device)
    critic2_target = SACCritic().to(device)
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    critic1_optimizer = torch.optim.Adam(critic1.parameters(), lr=lr)
    critic2_optimizer = torch.optim.Adam(critic2.parameters(), lr=lr)

    buffer = ReplayBuffer()
    episode_rewards = []
    episode_successes = []

    demo_buffer = ReplayBuffer(capacity=10 ** 9)
    demos = np.load(demos_path, allow_pickle=True)
    for traj in demos:
        lid_opened = grasped = False
        phis = []
        for step_d in traj:
            lid_now, hold_now = lidbox_env.stage_flags(step_d["state"])
            lid_opened = lid_opened or lid_now
            grasped = grasped or hold_now
            done_i = bool(np.ravel(step_d["done"])[0])
            phis.append(lidbox_env.SUCCESS_POTENTIAL if done_i else
                        lidbox_env.progress_potential(step_d["state"],
                                                      lid_opened, grasped))
        for i in range(len(traj) - 1):
            done_next = bool(np.ravel(traj[i + 1]["done"])[0])
            r = phis[i + 1] - phis[i] + (10.0 if done_next else 0.0)
            demo_buffer.push(traj[i]["state"], traj[i]["action"], r,
                             traj[i + 1]["state"], done_next)
    print(f"Loaded {len(demo_buffer)} demo transitions from {demos_path} "
          f"into a protected buffer (rewards recomputed as potential deltas)")

    def normalize_state(state):
        return (torch.from_numpy(state).float().to(device) - state_mean) / state_std

    def current_bc_lambda(episode_idx):
        if bc_lambda_anneal_episodes <= 0:
            return bc_lambda_final
        frac = min(1.0, episode_idx / bc_lambda_anneal_episodes)
        return bc_lambda + (bc_lambda_final - bc_lambda) * frac

    def current_online_mix(episode_idx):
        if online_mix_ramp_episodes <= 0:
            return online_mix_target
        frac = min(1.0, episode_idx / online_mix_ramp_episodes)
        return online_mix_target * frac

    diag = {"bc_loss": float("nan"), "q_scale": float("nan"),
            "bc_lambda": bc_lambda}

    def sample_mixed(online_frac=None):
        if online_frac is None:
            online_frac = 0.75
        n_online_target = int(round(batch_size * online_frac))
        n_online = min(n_online_target, len(buffer))
        n_demo = batch_size - n_online
        if n_online == 0:
            return demo_buffer.sample(n_demo)
        if n_demo == 0:
            return buffer.sample(n_online)
        d = demo_buffer.sample(n_demo)
        o = buffer.sample(n_online)
        return tuple(torch.cat([di, oi], dim=0) for di, oi in zip(d, o))

    def _gradient_step(batch, update_actor, bc_lambda_now=None):
        if bc_lambda_now is None:
            bc_lambda_now = bc_lambda
        states_b, actions_b, rewards_b, next_states_b, dones_b = batch
        states_b = states_b.to(device)
        actions_b = actions_b.to(device)
        rewards_b = rewards_b.to(device)
        next_states_b = next_states_b.to(device)
        dones_b = dones_b.to(device)

        states_b = (states_b - state_mean) / state_std
        next_states_b = (next_states_b - state_mean) / state_std

        with torch.no_grad():
            next_actions, next_log_probs, _, _ = actor.sample(next_states_b)
            q1_target = critic1_target(next_states_b, next_actions)
            q2_target = critic2_target(next_states_b, next_actions)
            q_target = torch.min(q1_target, q2_target) - alpha * next_log_probs
            q_target = rewards_b + gamma * (1.0 - dones_b) * q_target

        q1 = critic1(states_b, actions_b)
        q2 = critic2(states_b, actions_b)

        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)

        critic1_optimizer.zero_grad()
        critic1_loss.backward()
        critic1_optimizer.step()

        critic2_optimizer.zero_grad()
        critic2_loss.backward()
        critic2_optimizer.step()

        if update_actor:
            actor_loss, bc_loss, q_scale = compute_actor_loss(
                actor, critic1, critic2, bc_model, states_b,
                alpha=alpha, bc_lambda_now=bc_lambda_now)

            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_optimizer.step()

            diag["bc_loss"] = bc_loss.item()
            diag["q_scale"] = q_scale.item()
            diag["bc_lambda"] = bc_lambda_now

        for target_param, param in zip(critic1_target.parameters(), critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(critic2_target.parameters(), critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update(episode_for_lambda, online_mix_frac):
        batch = sample_mixed(online_mix_frac)
        _gradient_step(batch, update_actor=True,
                       bc_lambda_now=current_bc_lambda(episode_for_lambda))

    eval_rng = np.random.RandomState(eval_seed)
    eval_seeds = [int(eval_rng.randint(0, 2**31 - 1)) for _ in range(eval_episodes)]
    best_success = -1.0

    def eval_and_maybe_save(tag):
        nonlocal best_success
        rng_state = np.random.get_state()
        actor.eval()
        res = evaluate_policy(env, actor, bc_ckpt["state_mean"], bc_ckpt["state_std"],
                              seeds=eval_seeds, max_steps=env._max_timesteps,
                              device=device, deterministic=True, desc=f"eval {tag}")
        actor.train()
        np.random.set_state(rng_state)
        succ = res["success_rate"]
        improved = succ >= best_success
        print(f"\n[eval {tag}] success={succ:.2%} "
              f"(best={max(best_success, succ):.2%})"
              f"{'  <- new best, saved' if improved else ''}"
              f"  [diag bc_loss={diag['bc_loss']:.4f} "
              f"mean|Q|={diag['q_scale']:.1f} bc_lambda={diag['bc_lambda']:.2f}]",
              flush=True)
        if improved:
            best_success = succ
            torch.save({"actor_state_dict": actor.state_dict(),
                        "state_mean": bc_ckpt["state_mean"],
                        "state_std": bc_ckpt["state_std"]}, save_path)
        return succ

    eval_and_maybe_save("init")

    if warmup_steps > 0:
        for _ in tqdm(range(warmup_steps), desc="critic warmup"):
            _gradient_step(demo_buffer.sample(batch_size), update_actor=False)
        print(f"Critic warmup done ({warmup_steps} steps).", flush=True)

    if actor_freeze_episodes > 0:
        for ep in tqdm(range(actor_freeze_episodes), desc="actor freeze (collect)"):
            env.reset()
            state = env.high_level_state()
            done = False
            while not done:
                state_t = normalize_state(state).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, _ = actor.sample(state_t)
                action_np = action.cpu().numpy()[0]
                next_state, reward, done, truncated = env.step(action_np)
                buffer.push(state, action_np, reward, next_state, float(done))
                state = next_state
                if len(demo_buffer) + len(buffer) > batch_size:
                    _gradient_step(sample_mixed(current_online_mix(ep)),
                                   update_actor=False)
                if done or truncated:
                    break
        print(f"Actor-freeze done ({actor_freeze_episodes} eps, "
              f"online buffer={len(buffer)}).", flush=True)

    pbar = tqdm(range(num_episodes), desc="SAC training")
    for episode in pbar:
        env.reset()
        state = env.high_level_state()
        episode_reward = 0.0
        done = False

        while not done:
            state_t = normalize_state(state).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = actor.sample(state_t)
            action_np = action.cpu().numpy()[0]

            next_state, reward, done, truncated = env.step(action_np)
            buffer.push(state, action_np, reward, next_state, float(done))

            state = next_state
            episode_reward += reward

            if len(demo_buffer) + len(buffer) > batch_size:
                update(episode, current_online_mix(episode))

            if done or truncated:
                break

        episode_rewards.append(episode_reward)
        episode_successes.append(float(done))

        avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
        train_succ = np.mean(episode_successes[-50:]) if len(episode_successes) >= 50 else np.mean(episode_successes)
        pbar.set_postfix(avg_r=f"{avg_reward:.2f}", succ=f"{train_succ:.0%}",
                         buf=len(buffer))
        if (episode + 1) % 50 == 0:
            print(f"\n[ep {episode+1}] avg_reward(last50)={avg_reward:.2f}, "
                  f"train_success(last50)={train_succ:.0%}, "
                  f"buffer={len(buffer)}, bc_loss={diag['bc_loss']:.4f}, "
                  f"mean|Q|={diag['q_scale']:.1f}", flush=True)

        if (episode + 1) % ckpt_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"sac_ep{episode + 1:04d}.pt")
            torch.save({
                "actor_state_dict": actor.state_dict(),
                "state_mean": bc_ckpt["state_mean"],
                "state_std": bc_ckpt["state_std"],
            }, ckpt_path)

        if (episode + 1) % eval_every == 0:
            eval_and_maybe_save(f"ep{episode + 1}")

    eval_and_maybe_save("final")
    print(f"\nBest-by-success SAC policy (success={best_success:.2%}) saved to {save_path}", flush=True)

    del env
    return episode_rewards


def test_sac_smoke():
    train_sac(num_episodes=1, warmup_steps=100, actor_freeze_episodes=0,
              eval_episodes=2, ckpt_every=10**9, save_path="sac_smoke.pt",
              ckpt_dir="ckpt_smoke")
    assert os.path.exists("sac_smoke.pt")
    print("smoke OK")


if __name__ == "__main__":
    import sys
    if "--test-smoke" in sys.argv:
        test_sac_smoke()
    else:
        rewards = train_sac()
        print(f"Final 50-ep avg reward: {np.mean(rewards[-50:]):.2f}")
