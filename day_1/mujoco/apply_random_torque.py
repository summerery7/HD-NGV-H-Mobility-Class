import os
import time

import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mg400.xml")

ACTUATOR_NAMES = [
    "MG400_Joint_1",
    "MG400_Joint_2",
    "MG400_Joint_3",
    "M0G400_Joint_Gripper",
]

TORQUE_RATIO = 0.2
HOLD_TIME = 0.5


def sample_torque(model, act_ids, rng):
    lo = model.actuator_ctrlrange[act_ids, 0]
    hi = model.actuator_ctrlrange[act_ids, 1]
    return rng.uniform(lo, hi) * TORQUE_RATIO


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    act_ids = np.array([model.actuator(n).id for n in ACTUATOR_NAMES])
    rng = np.random.default_rng()

    tau = sample_torque(model, act_ids, rng)
    t_next = HOLD_TIME

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if data.time >= t_next:
                tau = sample_torque(model, act_ids, rng)
                t_next = data.time + HOLD_TIME
                print(f"t={data.time:6.2f}s  tau=[{tau[0]:7.2f} {tau[1]:7.2f} "
                      f"{tau[2]:7.2f} {tau[3]:7.2f}] Nm")

            data.ctrl[act_ids] = tau
            mujoco.mj_step(model, data)
            viewer.sync()

            dt_left = model.opt.timestep - (time.time() - step_start)
            if dt_left > 0:
                time.sleep(dt_left)


if __name__ == "__main__":
    main()
