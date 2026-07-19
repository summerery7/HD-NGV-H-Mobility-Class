import os
import time

import mujoco
import mujoco.viewer

XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mg400.xml")


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"model: {XML_PATH}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        lo, hi = model.actuator_ctrlrange[i]
        print(f"  actuator[{i}] {name}: ctrlrange=({lo:.0f}, {hi:.0f})")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            dt_left = model.opt.timestep - (time.time() - step_start)
            if dt_left > 0:
                time.sleep(dt_left)


if __name__ == "__main__":
    main()
