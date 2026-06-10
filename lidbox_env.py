import numpy as np
import torch
import torchvision.transforms as transforms
import environment

LID_OPEN_ANGLE = 0.5      
GRASP_GRIP_MIN = 0.3      
GRASP_DIST_MAX = 0.08     
LIFT_HEIGHT_MIN = 0.02    
TABLE_Z = 1.05
SUCCESS_POTENTIAL = 4.0


def _prox(dist):
    return 1.0 / (1.0 + 10.0 * dist)


def stage_flags(state):
    state = np.asarray(state, dtype=np.float64)
    ee, sphere = state[0:3], state[3:6]
    lid_open = state[9] > LID_OPEN_ANGLE
    holding = (state[10] > GRASP_GRIP_MIN
               and np.linalg.norm(ee - sphere) < GRASP_DIST_MAX
               and (sphere[2] - TABLE_Z) > LIFT_HEIGHT_MIN)
    return bool(lid_open), bool(holding)


def progress_potential(state, lid_opened, grasped):
    state = np.asarray(state, dtype=np.float64)
    ee, sphere, box = state[0:3], state[3:6], state[6:9]
    lid_angle = state[9]
    if not lid_opened:
        lid_front_xy = box[:2] + np.array([0.0, 0.05])
        d_lid = np.linalg.norm(ee[:2] - lid_front_xy)
        return 0.5 * _prox(d_lid) + 0.5 * float(np.clip(lid_angle / LID_OPEN_ANGLE, 0.0, 1.0))
    if not grasped:
        return 1.0 + _prox(np.linalg.norm(ee - sphere))
    return 2.0 + _prox(np.linalg.norm(sphere[:2] - box[:2]))


class LidBoxEnv(environment.BaseEnv):
    def __init__(self, render_mode="gui"):
        self.viewer = None
        self._render_mode = render_mode
        self._delta = 0.05
        self._max_timesteps = 120
        self._goal_thresh = 0.02
        super().__init__(render_mode=render_mode)

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        scene = environment.create_tabletop_scene()

        # random pos for sphere
        sphere_xy = [np.random.uniform(0.40, 0.65), np.random.uniform(-0.20, 0.20)]
        environment.create_object(scene, "sphere", pos=[sphere_xy[0], sphere_xy[1], 1.05],
                                  quat=[0, 0, 0, 1], size=[0.03],
                                  rgba=[0.2, 0.8, 0.2, 1], name="sphere")

        # lidded box near the edge of the table
        box_xy = [0.45, 0.25]
        box_size = [0.04, 0.04, 0.04]
        environment.create_box(scene, pos=[box_xy[0], box_xy[1], 1.02],
                               quat=[0, 0, 0, 1], size=box_size, width=0.005,
                               rgba=[0.8, 0.5, 0.2, 1], lid_type="hinge", name="box",
                               static=True)

        return scene

    def high_level_state(self):
        ee = self.data.site(self._ee_site).xpos.copy()
        sphere = self.data.body("sphere").xpos.copy()
        box_body = self.data.body("box").xpos.copy()
        
        lid_joint_id = self.model.joint("lid_joint").qposadr  # lid angle
        lid_angle = self.data.qpos[lid_joint_id]
        
        gripper_pos = self._get_joint_position()[6]  # gripper position
        return np.array([
            float(ee[0]), float(ee[1]), float(ee[2]),
            float(sphere[0]), float(sphere[1]), float(sphere[2]),
            float(box_body[0]), float(box_body[1]), float(box_body[2]),
            float(lid_angle), float(gripper_pos)
        ], dtype=np.float32)

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels.float() / 255.0

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action = np.asarray(action, dtype=np.float64)

        delta_xyz = action[:3] * self._delta
        ee_pos = self.data.site(self._ee_site).xpos.copy()
        target_xyz = ee_pos + delta_xyz

        target_xyz[0] = np.clip(target_xyz[0], 0.30, 0.70)
        target_xyz[1] = np.clip(target_xyz[1], -0.30, 0.50)
        target_xyz[2] = np.clip(target_xyz[2], 1.02, 1.35)

        # [-1,1] -> [0, 1] (0=open, 1=closed)
        gripper_cmd = float((action[3] + 1.0) / 2.0)
        gripper_cmd = np.clip(gripper_cmd, 0.0, 1.0)

        self._set_ee_in_cartesian(target_xyz, rotation=[-90, 0, 180],
                                  n_splits=60, threshold=0.02)

        gripper_joint_val = gripper_cmd * 0.8
        self._set_joint_position({6: gripper_joint_val}, max_iters=200)

        self._t += 1

        done = bool(self._sphere_in_box())
        truncated = self._t >= self._max_timesteps
        state = self.high_level_state()

        lid_now, hold_now = stage_flags(state)
        self._lid_opened = self._lid_opened or lid_now
        self._grasped = self._grasped or hold_now
        phi_new = SUCCESS_POTENTIAL if done else progress_potential(
            state, self._lid_opened, self._grasped)
        prev_phi = self._prev_phi if self._prev_phi is not None else phi_new
        reward = phi_new - prev_phi + (10.0 if done else 0.0)
        self._prev_phi = phi_new

        return state, reward, done, truncated

    def _compute_reward(self):
        ee = self.data.site(self._ee_site).xpos.copy()
        sphere = self.data.body("sphere").xpos.copy()
        box = self.data.body("box").xpos.copy()
        lid_joint_id = self.model.joint("lid_joint").qposadr
        lid_angle = float(self.data.qpos[lid_joint_id])
        gripper_val = self._get_joint_position()[6]

        # approach lid
        lid_front = box.copy()
        lid_front[1] += 0.05
        lid_front[2] += 0.06
        dist_to_lid = max(np.linalg.norm(ee[:2] - lid_front[:2]), 0.001)
        r_approach_lid = 1.0 / (1.0 + 10.0 * dist_to_lid)

        # open lid
        r_open_lid = lid_angle / np.pi 

        # approach sphere
        dist_to_sphere = max(np.linalg.norm(ee - sphere), 0.001)
        r_approach_obj = 1.0 / (1.0 + 10.0 * dist_to_sphere)

        # grasp
        ee_to_sphere_dist = np.linalg.norm(ee - sphere)
        sphere_height = sphere[2] - 1.05
        r_grasp = 0.0
        if gripper_val > 0.3 and ee_to_sphere_dist < 0.08 and sphere_height > 0.02:
            r_grasp = 2.0

        # transport to box
        sphere_xy = sphere[:2]
        box_xy = box[:2]
        dist_to_box = max(np.linalg.norm(sphere_xy - box_xy), 0.001)
        r_transport = 1.0 / (1.0 + 10.0 * dist_to_box) if sphere_height > 0.02 else 0.0

        # terminal bonus
        r_terminal = 10.0 if self._sphere_in_box() else 0.0

        return r_approach_lid + r_open_lid + r_approach_obj + r_grasp + r_transport + r_terminal

    def _sphere_in_box(self):
        sphere = self.data.body("sphere").xpos.copy()
        box = self.data.body("box").xpos.copy()
        lid_angle = self.data.qpos[self.model.joint("lid_joint").qposadr]

        box_top = box[2] + 0.04
        in_xy = abs(sphere[0] - box[0]) < 0.035 and abs(sphere[1] - box[1]) < 0.035
        in_z = (sphere[2] > box_top - 0.04) and (sphere[2] < box_top + 0.02)
        lid_open = lid_angle > LID_OPEN_ANGLE
        return bool(in_xy and in_z and lid_open)

    def reset(self):
        super().reset()
        self._t = 0
        self._lid_opened = False
        self._grasped = False
        self._prev_phi = (progress_potential(self.high_level_state(), False, False)
                          if hasattr(self, "_ee_site") else None)


if __name__ == "__main__":
    env = LidBoxEnv(render_mode="gui")
    for episode in range(5):
        env.reset()
        cumulative_reward = 0.0
        for step in range(50):
            action = np.random.uniform(-1, 1, 4)
            action[3] = -1.0
            state, reward, done, truncated = env.step(action)
            cumulative_reward += reward
            if done or truncated:
                break
        print(f"[INFO] Episode {episode}: reward={cumulative_reward:.2f}")
