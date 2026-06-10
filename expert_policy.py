import numpy as np

OPEN, CLOSE = -1.0, 1.0

LID_OPEN_THRESH = 1.0      
GRIP_CLOSED_THRESH = 0.4   
CLEAR_Z = 1.30             
CROSS_Y_OFFSET = -0.10     
GRASP_DIST = 0.08          


def _toward(ee, target, grip, delta):
    target = np.asarray(target, dtype=np.float64)
    move = np.clip((target - ee) / delta, -1.0, 1.0)
    return np.array([move[0], move[1], move[2], grip], dtype=np.float32)


def expert_action(env):
    ee = env.data.site(env._ee_site).xpos.copy()
    sphere = env.data.body("sphere").xpos.copy()
    box = env.data.body("box").xpos.copy()
    lid_angle = float(np.ravel(env.data.qpos[env.model.joint("lid_joint").qposadr])[0])
    grip = float(env._get_joint_position()[6])
    delta = env._delta
    hx, hy = float(box[0]), float(box[1])

    lid_open = lid_angle > LID_OPEN_THRESH
    grip_closed = grip > GRIP_CLOSED_THRESH
    ee_to_sphere = float(np.linalg.norm(ee - sphere))
    sphere_grasped = grip_closed and ee_to_sphere < GRASP_DIST

    # lid still closed
    if not lid_open:
        if not grip_closed:
            xy_aligned = np.linalg.norm(ee[:2] - np.array([hx, hy])) < 0.03
            if not xy_aligned:
                return _toward(ee, [hx, hy, max(float(ee[2]), 1.18)], OPEN, delta)
            if ee[2] > 1.12:
                return _toward(ee, [hx, hy, 1.10], OPEN, delta)   
            return _toward(ee, [hx, hy, 1.10], CLOSE, delta)      
        return _toward(ee, [hx, hy + 0.20, 1.15], CLOSE, delta)

    # lid open
    if not sphere_grasped:
        if grip_closed:
            return _toward(ee, [float(ee[0]), float(ee[1]), CLEAR_Z], OPEN, delta)
        on_approach_side = ee[1] <= hy + CROSS_Y_OFFSET + 0.05
        if ee[2] < CLEAR_Z - 0.04 and not on_approach_side:
            return _toward(ee, [hx, float(ee[1]), CLEAR_Z], OPEN, delta)
        if not on_approach_side:
            return _toward(ee, [hx, hy + CROSS_Y_OFFSET, CLEAR_Z], OPEN, delta)
        above_sphere = sphere + np.array([0.0, 0.0, 0.10])
        if (np.linalg.norm(ee[:2] - above_sphere[:2]) > 0.03) or (ee[2] > above_sphere[2] + 0.05):
            return _toward(ee, above_sphere, OPEN, delta)
        if ee_to_sphere > 0.04:
            return _toward(ee, sphere + np.array([0.0, 0.0, -0.01]), OPEN, delta)
        return _toward(ee, sphere + np.array([0.0, 0.0, -0.01]), CLOSE, delta)  

    # sphere grasped
    if ee[2] < 1.20:
        return _toward(ee, [float(ee[0]), float(ee[1]), 1.22], CLOSE, delta)  
    if np.linalg.norm(ee[:2] - np.array([hx, hy])) > 0.03:
        return _toward(ee, [hx, hy, 1.22], CLOSE, delta)                     
    return _toward(ee, [hx, hy, 1.22], OPEN, delta)                         
