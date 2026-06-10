import argparse
import glob
import json
import os
import re

import numpy as np
import torch

import lidbox_env
from eval import evaluate_policy
from models import BCPolicy, SACGaussianPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=591)
    parser.add_argument("--out", default="checkpoint_sweep.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = lidbox_env.LidBoxEnv(render_mode="offscreen")
    max_steps = env._max_timesteps

    rng = np.random.RandomState(args.seed)
    seeds = [int(rng.randint(0, 2**31 - 1)) for _ in range(args.episodes)]  # rand seed gen

    results = {"episodes": args.episodes, "seed": args.seed, "sweep": []}

    ckpt = torch.load("bc_policy.pt", map_location=device, weights_only=False)
    bc = BCPolicy().to(device)
    bc.load_state_dict(ckpt["model_state_dict"])
    bc.eval()
    r = evaluate_policy(env, bc, ckpt["state_mean"], ckpt["state_std"], seeds=seeds, max_steps=max_steps, device=device, desc="BC")
    results["bc"] = {k: r[k] for k in ("success_rate", "mean_reward", "std_reward", "stage_completion", "success_list")}
    print(f"[BC] success={r['success_rate']:.2%}")

    paths = sorted(glob.glob("checkpoints/sac_ep*.pt"))
    for path in paths:
        episode = int(re.search(r"sac_ep(\d+)\.pt", path).group(1))
        ckpt = torch.load(path, map_location=device, weights_only=False)
        sac = SACGaussianPolicy().to(device)
        sac.load_state_dict(ckpt["actor_state_dict"])
        sac.eval()
        r = evaluate_policy(env, sac, ckpt["state_mean"], ckpt["state_std"], seeds=seeds, max_steps=max_steps, device=device, deterministic=True, desc=f"ep{episode}")
        results["sweep"].append({
            "episode": episode,
            "success_rate": r["success_rate"],
            "mean_reward": r["mean_reward"],
            "std_reward": r["std_reward"],
            "stage_completion": r["stage_completion"],
            "success_list": r["success_list"],
        })
        print(f"[ep{episode:4d}] success={r['success_rate']:.2%}")
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)

    print(f"Sweep done: {len(paths)} checkpoints -> {args.out}")
    del env


if __name__ == "__main__":
    main()
