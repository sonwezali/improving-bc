import numpy as np
from tqdm import tqdm

import lidbox_env

OPEN = -1.0   # gripper open
CLOSE = 1.0   # gripper closed


def record_step(env, action, trajectory):
    trajectory.append({
        "state": env.high_level_state(),  # using high-level states is easier to implement
        "action": np.asarray(action, dtype=np.float32),
        "reward": env._compute_reward(),
        "done": env._sphere_in_box(),
    })


def move_to(env, target, grip, trajectory, thresh=0.02, max_steps=25):
    target = np.asarray(target, dtype=np.float64)
    for _ in range(max_steps):
        ee = env.data.site(env._ee_site).xpos.copy()
        delta = target - ee
        if np.linalg.norm(delta) < thresh:
            break
        move = np.clip(delta / env._delta, -1.0, 1.0)
        action = np.array([move[0], move[1], move[2], grip], dtype=np.float32)
        record_step(env, action, trajectory)
        env.step(action)


def actuate_gripper(env, grip, trajectory, n_settle=6):
    action = np.array([0.0, 0.0, 0.0, grip], dtype=np.float32)
    for _ in range(n_settle):
        record_step(env, action, trajectory)
        env.step(action)


def collect_demo(env, render=False):
    trajectory = []
    box_xy = np.array([0.45, 0.25])
    handle_xy = np.array([box_xy[0], box_xy[1]])

    # grip the lid
    move_to(env, [handle_xy[0], handle_xy[1], 1.18], OPEN, trajectory)   
    move_to(env, [handle_xy[0], handle_xy[1], 1.10], OPEN, trajectory)   
    actuate_gripper(env, CLOSE, trajectory, n_settle=6)                  

    # open the lid
    move_to(env, [handle_xy[0], handle_xy[1], 1.12], CLOSE, trajectory)         
    move_to(env, [handle_xy[0], handle_xy[1] + 0.20, 1.15], CLOSE, trajectory)  
    actuate_gripper(env, OPEN, trajectory, n_settle=4)                          

    move_to(env, [handle_xy[0], handle_xy[1] + 0.18, 1.30], OPEN, trajectory)  # move up not to close the lid again
    move_to(env, [handle_xy[0], handle_xy[1] - 0.10, 1.30], OPEN, trajectory)  # cross the lid

    # approach the sphere
    sphere = env.data.body("sphere").xpos.copy()
    move_to(env, sphere + np.array([0.0, 0.0, 0.10]), OPEN, trajectory)   # diagonal scoop above sphere
    move_to(env, sphere + np.array([0.0, 0.0, -0.01]), OPEN, trajectory)  # this is just an experimental value because the sphere
                                                                          # slipped repeatedly before adding this
    actuate_gripper(env, CLOSE, trajectory, n_settle=10)                  

    # transport over the box opening and drop the sphere in
    move_to(env, sphere + np.array([0.0, 0.0, 0.12]), CLOSE, trajectory)  
    move_to(env, [box_xy[0], box_xy[1], 1.22], CLOSE, trajectory)         
    actuate_gripper(env, OPEN, trajectory, n_settle=6)                    

    record_step(env, np.array([0.0, 0.0, 0.0, OPEN], dtype=np.float32), trajectory)
    return trajectory


def collect_demos(num_demos=300, seed=591, only_successful=True):
    np.random.seed(seed)
    all_trajs = []
    n_success = 0
    env       = lidbox_env.LidBoxEnv(render_mode="offscreen")
    pbar      = tqdm(range(num_demos), desc="Collecting demos")

    for i in pbar:
        env.reset()
        traj = collect_demo(env)
        success = bool(traj[-1]["done"])
        n_success += success
        if success or not only_successful:
            all_trajs.append(traj)
        pbar.set_postfix(kept=len(all_trajs), success=n_success)
    del env
    print(f"[INFO] Done: {n_success}/{num_demos} demos succeeded and kept {len(all_trajs)}")  # to track how many of the demos succeeded
    return all_trajs


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "collect":
        num = int(sys.argv[2]) if len(sys.argv) > 2 else 300
        trajs = collect_demos(num_demos=num)
        np.save("demos.npy", np.array(trajs, dtype=object), allow_pickle=True)
        n_steps = sum(len(t) for t in trajs)
        print(f"[INFO] Saved {len(trajs)} demos ({n_steps} transitions) to demos.npy")
    else:
        env = lidbox_env.LidBoxEnv(render_mode="gui")
        env.reset()
        traj = collect_demo(env, render=True)
        print(f"[INFO] Trajectory length: {len(traj)} steps")
        print(f"[INFO] Final reward: {traj[-1]['reward']:.2f}")
        print(f"[INFO] Done: {traj[-1]['done']}")
