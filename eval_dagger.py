import argparse
import glob
import json
import re

import numpy as np
import torch

import lidbox_env
from eval import evaluate_policy
from models import BCPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=591)
    parser.add_argument("--baseline", default=None,
                        help="path to the pre-DAgger bc_policy_baseline.pt")
    parser.add_argument("--out", default="dagger_sweep.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = lidbox_env.LidBoxEnv(render_mode="offscreen")
    max_steps = env._max_timesteps

    rng = np.random.RandomState(args.seed)
    seeds = [int(rng.randint(0, 2**31 - 1)) for _ in range(args.episodes)]

    targets = []
    if args.baseline:
        targets.append((0, args.baseline))
    for path in sorted(glob.glob("checkpoints_dagger/bc_dagger_iter*.pt")):
        targets.append((int(re.search(r"iter(\d+)\.pt", path).group(1)), path))

    results = {"episodes": args.episodes, "seed": args.seed, "sweep": []}
    for iteration, path in targets:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        bc = BCPolicy().to(device)
        bc.load_state_dict(ckpt["model_state_dict"])
        bc.eval()
        r = evaluate_policy(env, bc, ckpt["state_mean"], ckpt["state_std"], seeds=seeds, max_steps=max_steps, device=device, desc=f"iter{iteration}")
        results["sweep"].append({
            "iteration": iteration,
            "success_rate": r["success_rate"],
            "mean_reward": r["mean_reward"],
            "std_reward": r["std_reward"],
            "stage_completion": r["stage_completion"],
            "success_list": r["success_list"],
        })
        print(f"[iter{iteration}] success={r['success_rate']:.2%}")
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)

    print(f"DAgger sweep done: {len(targets)} policies -> {args.out}")
    del env


if __name__ == "__main__":
    main()
