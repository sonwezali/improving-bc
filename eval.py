import os
import numpy as np
import torch
from tqdm import tqdm
import lidbox_env
from models import BCPolicy, SACGaussianPolicy


def evaluate_policy(env, model=None, state_mean=None, state_std=None,
                    num_episodes=50, max_steps=80, device="cpu", render=False,
                    deterministic=False, seeds=None, desc="evaluating"):
    if seeds is not None:
        num_episodes = len(seeds)

    successes     = 0
    rewards       = []
    success_list  = []
    stage_reached = {"approach_lid": 0, "lid_open": 0, "approach_sphere": 0, "grasp": 0, "transport": 0, "drop": 0}

    for ep in tqdm(range(num_episodes), desc=desc, leave=False):
        if seeds is not None:
            np.random.seed(seeds[ep])   # same scene across policies to guarantee fair comparison
        env.reset()
        episode_reward    = 0.0
        lid_was_open      = False
        sphere_was_lifted = False
        done              = False
        state             = env.high_level_state()

        for step in range(max_steps):
            if model is not None:
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                if state_mean is not None:
                    state_t = (state_t - torch.from_numpy(state_mean).to(device)) / torch.from_numpy(state_std).to(device)
                with torch.no_grad():
                    if deterministic and hasattr(model, 'deterministic_action'):  # greedy
                        action = model.deterministic_action(state_t).cpu().numpy()[0]
                    else:
                        action = model(state_t).cpu().numpy()[0]
            else:
                action = np.random.uniform(-1, 1, 4)

            state, reward, done, truncated = env.step(action)
            episode_reward += reward

            lid_angle = state[9]
            sphere_z = state[5]
            if lid_angle > 0.3:
                lid_was_open = True
            if sphere_z > 1.08:
                sphere_was_lifted = True

            if done or truncated:
                break

        success_list.append(bool(done))
        if done:
            successes                        += 1
            stage_reached["drop"]            += 1
            stage_reached["transport"]       += 1
            stage_reached["grasp"]           += 1
            stage_reached["approach_sphere"] += 1
            stage_reached["lid_open"]        += 1
            stage_reached["approach_lid"]    += 1
        else:
            if sphere_was_lifted:
                stage_reached["transport"]       += 1
                stage_reached["grasp"]           += 1
                stage_reached["approach_sphere"] += 1
            if lid_was_open:
                stage_reached["lid_open"]     += 1
                stage_reached["approach_lid"] += 1
            else:
                ee = state[:3]
                box = state[6:9]
                if np.linalg.norm(ee[:2] - box[:2]) < 0.1:
                    stage_reached["approach_lid"] += 1

        rewards.append(episode_reward)

    return {
        "success_rate": successes / num_episodes,
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "stage_completion": {k: v / num_episodes for k, v in stage_reached.items()},
        "success_list": success_list,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10,
                        help="episodes per policy (paired on the same scenes)")
    parser.add_argument("--seed", type=int, default=591,
                        help="base seed for the shared paired scene set")
    parser.add_argument("--no-random", action="store_true",
                        help="skip the random-policy baseline")
    parser.add_argument("--no-sac", action="store_true",
                        help="skip the SAC policy (e.g. when testing BC alone)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = lidbox_env.LidBoxEnv(render_mode="offscreen")
    max_steps = env._max_timesteps
    rng = np.random.RandomState(args.seed)
    seeds = [int(rng.randint(0, 2**31 - 1)) for _ in range(args.episodes)]

    results = {}

    if not args.no_random:
        print("Evaluating random policy...")
        results["random"] = evaluate_policy(env, model=None, seeds=seeds,
                                            max_steps=max_steps, device=device,
                                            desc="random")

    if os.path.exists("bc_policy.pt"):
        ckpt = torch.load("bc_policy.pt", map_location=device, weights_only=False)
        bc = BCPolicy().to(device)
        bc.load_state_dict(ckpt["model_state_dict"])
        bc.eval()
        print("Evaluating BC policy...")
        results["bc"] = evaluate_policy(env, bc, ckpt["state_mean"], ckpt["state_std"],
                                        seeds=seeds, max_steps=max_steps, device=device,
                                        desc="BC")
    else:
        print("[WARN] bc_policy.pt not found - skipping BC.")

    if not args.no_sac and os.path.exists("sac_policy.pt"):
        ckpt = torch.load("sac_policy.pt", map_location=device, weights_only=False)
        sac = SACGaussianPolicy().to(device)
        sac.load_state_dict(ckpt["actor_state_dict"])
        sac.eval()
        print("Evaluating BC+SAC policy...")
        results["sac"] = evaluate_policy(env, sac, ckpt["state_mean"], ckpt["state_std"],
                                         seeds=seeds, max_steps=max_steps, device=device,
                                         deterministic=True, desc="BC+SAC")
    else:
        print("[WARN] sac_policy.pt not found - skipping SAC (train_sac.py to compare).")

    for name, r in results.items():
        label = {"bc": "BC", "sac": "BC+SAC", "random": "RANDOM"}.get(name, name.upper())
        print(f"\n{label}:")
        print(f"  Success: {r['success_rate']:.2%}")
        print(f"  Reward: {r['mean_reward']:.2f} ± {r['std_reward']:.2f}")
        for stage, rate in r["stage_completion"].items():
            print(f"    {stage}: {rate:.2%}")

    if "bc" in results and "sac" in results:
        bc_only = sac_only = ties = 0
        for b, s in zip(results["bc"]["success_list"], results["sac"]["success_list"]):
            if b and not s:
                bc_only += 1
            elif s and not b:
                sac_only += 1
            else:
                ties += 1
        print("\nPaired BC vs BC+SAC (same scenes):")
        print(f"  only BC succeeded:     {bc_only}")
        print(f"  only BC+SAC succeeded: {sac_only}")
        print(f"  ties:                  {ties}")

    del env
