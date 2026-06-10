import sys
import torch
import numpy as np
import lidbox_env
from models import BCPolicy, SACGaussianPolicy


def _load_policy(policy_path, policy_type, device):
    ckpt = torch.load(policy_path, map_location=device, weights_only=False)
    state_mean = torch.from_numpy(ckpt["state_mean"]).float().to(device)
    state_std = torch.from_numpy(ckpt["state_std"]).float().to(device)
    if policy_type == "sac":
        model = SACGaussianPolicy().to(device)
        model.load_state_dict(ckpt["actor_state_dict"])
    else:
        model = BCPolicy().to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, state_mean, state_std


def _run_episode(env, model, state_mean, state_std, policy_type, device, deterministic=True):
    env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    while not done:
        state = env.high_level_state()
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        state_t = (state_t - state_mean) / state_std
        with torch.no_grad():
            if deterministic and policy_type == "sac":
                action = model.deterministic_action(state_t).cpu().numpy()[0]
            else:
                action = model(state_t).cpu().numpy()[0]
        _, reward, done, truncated = env.step(action)
        total_reward += reward
        steps += 1
        if truncated:
            break
    return total_reward, steps, done


def run_demo(policy_path, policy_type="bc", deterministic=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state_mean, state_std = _load_policy(policy_path, policy_type, device)
    env = lidbox_env.LidBoxEnv(render_mode="gui")
    reward, steps, success = _run_episode(env, model, state_mean, state_std, policy_type, device, deterministic)
    print(f"[{policy_type.upper()}] steps={steps}, reward={reward:.2f}, success={success}")


def run_comparison(bc_path="bc_policy.pt", sac_path="sac_policy.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bc_model, bc_mean, bc_std = _load_policy(bc_path, "bc", device)
    sac_model, sac_mean, sac_std = _load_policy(sac_path, "sac", device)

    env = lidbox_env.LidBoxEnv(render_mode="gui")

    print("--- Episode 1: BC policy ---")
    reward, steps, success = _run_episode(env, bc_model, bc_mean, bc_std, "bc", device)
    print(f"[BC]  steps={steps}, reward={reward:.2f}, success={success}")

    input("Press Enter to start SAC episode...")

    print("--- Episode 2: SAC policy ---")
    reward, steps, success = _run_episode(env, sac_model, sac_mean, sac_std, "sac", device)
    print(f"[SAC] steps={steps}, reward={reward:.2f}, success={success}")


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "compare":
        run_comparison()
    else:
        policy_path = sys.argv[1]
        policy_type = sys.argv[2] if len(sys.argv) > 2 else "bc"
        run_demo(policy_path, policy_type)
