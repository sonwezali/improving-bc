import os
import numpy as np
import torch
from tqdm import tqdm

import lidbox_env
from models import BCPolicy
from train_bc import fit_bc, _load_transitions
from expert_policy import expert_action
from eval import evaluate_policy


def _beta(iteration, num_iters):
    if num_iters <= 1:
        return 0.0
    return max(0.0, 1.0 - iteration / (num_iters - 1))


def _policy_action(model, state, state_mean, state_std, device):
    s = torch.from_numpy(state).float().unsqueeze(0).to(device)
    s = (s - torch.from_numpy(state_mean).to(device)) / torch.from_numpy(state_std).to(device)
    with torch.no_grad():
        return model(s).cpu().numpy()[0]


def train_dagger(demos_path="demos.npy", num_iters=6, rollouts_per_iter=25,
                 bc_epochs=100, eval_episodes=40, eval_seed=591,
                 ckpt_dir="checkpoints_dagger", save_path="bc_policy.pt", seed=591):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(ckpt_dir, exist_ok=True)
    np.random.seed(seed)

    env = lidbox_env.LidBoxEnv(render_mode="offscreen")

    states, actions = _load_transitions(demos_path)
    agg_states = [s for s in states]
    agg_actions = [a for a in actions]
    print(f"Seeded DAgger dataset with {len(agg_states)} demo transitions")

    rng = np.random.RandomState(eval_seed)
    eval_seeds = [int(rng.randint(0, 2**31 - 1)) for _ in range(eval_episodes)]
    max_steps = env._max_timesteps

    rollout_rng = np.random.RandomState(seed)

    history = {"success": []}
    best_success = -1.0

    for it in range(num_iters):
        model, state_mean, state_std = fit_bc(
            np.array(agg_states, dtype=np.float32),
            np.array(agg_actions, dtype=np.float32),
            epochs=bc_epochs, device=device, verbose=False)
        model.eval()

        beta = _beta(it, num_iters)
        new_labels = 0
        for _ in tqdm(range(rollouts_per_iter), desc=f"DAgger iter {it+1}/{num_iters} (beta={beta:.2f})"):
            np.random.seed(int(rollout_rng.randint(0, 2**31 - 1)))
            env.reset()
            state = env.high_level_state()
            for _ in range(max_steps):
                a_expert = expert_action(env)
                agg_states.append(state.copy())
                agg_actions.append(a_expert.copy())
                new_labels += 1
                if rollout_rng.rand() < beta:
                    a_exec = a_expert
                else:
                    a_exec = _policy_action(model, state, state_mean, state_std, device)
                state, _, done, truncated = env.step(a_exec)
                if done or truncated:
                    break

        ckpt = {"model_state_dict": model.state_dict(),
                "state_mean": state_mean, "state_std": state_std}
        torch.save(ckpt, os.path.join(ckpt_dir, f"bc_dagger_iter{it+1:02d}.pt"))

        res = evaluate_policy(env, model, state_mean, state_std, seeds=eval_seeds,
                              max_steps=max_steps, device=device, desc=f"eval iter {it+1}")
        succ = res["success_rate"]
        history["success"].append(succ)
        print(f"\n[DAgger iter {it+1}] beta={beta:.2f}  success={succ:.2%}  "
              f"dataset={len(agg_states)}  (+{new_labels} labels)  "
              f"stages={res['stage_completion']}", flush=True)

        if succ >= best_success:
            best_success = succ
            torch.save(ckpt, save_path)

    del env
    history["best_success"] = best_success
    print(f"\nDAgger done. Best success={best_success:.2%}; best policy saved to {save_path}")
    return history


if __name__ == "__main__":
    train_dagger()
