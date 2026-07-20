#!/usr/bin/env python3

import os
import sys
import threading
import time
from collections import deque

import numpy as np
import mujoco
import mujoco.viewer

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray
from dobot_msgs.msg import JointCmd, PoseCmd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import computed_torque_control as ctc
from cubic_trajectory import CubicTrajectory
from simple_fk import forward_kinematics
from simple_ik import inverse_kinematics
from simple_jac import jacobian

DEFAULT_JOINT_SPEED = 30.0
DEFAULT_LINEAR_SPEED = 50.0
STATE_PUB_EVERY = 20


class MujocoControllerNode(Node):
    def __init__(self):
        super().__init__("mujoco_controller")

        self.model = ctc.load_model()
        self.base, self.I_extra = ctc.setup_controller(self.model)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = ctc.dt_physics
        self.n_sub = int(round(ctc.dt / ctc.dt_physics))

        self.qadr = np.array([self.model.jnt_qposadr[self.model.joint(n).id]
                              for n in ctc.joint_names])
        self.dadr = np.array([self.model.jnt_dofadr[self.model.joint(n).id]
                              for n in ctc.joint_names])
        self.act_ids = np.array([self.model.actuator(n).id
                                 for n in ctc.actuator_names])
        self.force_lo, self.force_hi = \
            self.model.actuator_forcerange[self.act_ids].T

        mujoco.mj_forward(self.model, self.data)


        self.lock = threading.Lock()
        self.queue = deque()
        self.traj = None
        self.hold_q = self.data.qpos[self.qadr].copy()

        self.create_subscription(JointCmd, "mujoco/cmd/joint", self.cb_joint, 10)
        self.create_subscription(PoseCmd, "mujoco/cmd/mov_l", self.cb_mov_l, 10)
        self.pub_js = self.create_publisher(JointState, "mujoco/joint_states", 10)
        self.pub_pose = self.create_publisher(Float64MultiArray, "mujoco/pose", 10)
        self.pub_moving = self.create_publisher(Bool, "mujoco/moving", 10)

        self.get_logger().info("mujoco_controller 준비 완료 — "
                               "mujoco/cmd/joint, mujoco/cmd/mov_l 대기 중")


    def _tail_q(self):
        if self.queue:
            return self.queue[-1]["q"][-1].copy()
        if self.traj is not None:
            return self.traj["q"][-1].copy()
        return self.hold_q.copy()

    def cb_joint(self, msg: JointCmd):
        q1 = np.deg2rad([msg.j1, msg.j2, msg.j3, msg.j4])
        speed = msg.speed if msg.speed > 0 else DEFAULT_JOINT_SPEED
        with self.lock:
            q0 = self._tail_q()
            T = max(float(np.max(np.abs(q1 - q0))) * ctc.R2D / speed, 0.8)
            traj = self._plan_joint(q0, q1, T)
            traj["label"] = (f"joint→({msg.j1:.1f}, {msg.j2:.1f}, "
                             f"{msg.j3:.1f}, {msg.j4:.1f})deg, {T:.1f}s")
            self.queue.append(traj)
        self.get_logger().info(f"명령 수신: {traj['label']}")

    def cb_mov_l(self, msg: PoseCmd):
        p1_w = np.array([msg.x, msg.y, msg.z]) / 1000.0
        r1 = np.deg2rad(msg.r)


        th = inverse_kinematics(ctc.world2dh_p(p1_w, self.base))
        p_chk = ctc.dh2world_p(forward_kinematics(np.append(th, 0.0))[0], self.base)
        if np.linalg.norm(p_chk - p1_w) > 1e-3:
            self.get_logger().error(
                f"도달 불가 목표 (x={msg.x:.1f}, y={msg.y:.1f}, z={msg.z:.1f})mm — 무시")
            return

        speed = msg.speed if msg.speed > 0 else DEFAULT_LINEAR_SPEED
        with self.lock:
            q0 = self._tail_q()
            p0_w = ctc.dh2world_p(forward_kinematics(ctc.mj2dh_q(q0))[0], self.base)
            dist_mm = float(np.linalg.norm(p1_w - p0_w)) * 1000.0
            T = max(dist_mm / speed, abs(r1 - q0[3]) * ctc.R2D / DEFAULT_JOINT_SPEED, 1.0)
            traj = self._plan_cartesian(q0, p0_w, p1_w, r1, T)
            traj["label"] = (f"mov_l→({msg.x:.1f}, {msg.y:.1f}, {msg.z:.1f})mm, "
                             f"{dist_mm:.0f}mm/{T:.1f}s")
            self.queue.append(traj)
        self.get_logger().info(f"명령 수신: {traj['label']}")


    def _plan_joint(self, q0, q1, T):
        traj = CubicTrajectory([q0, q1], [T])
        n = int(round(T / ctc.dt)) + 1
        q_tab = np.zeros((n, 4)); v_tab = np.zeros((n, 4)); a_tab = np.zeros((n, 4))
        for k in range(n):
            q_tab[k], v_tab[k], a_tab[k] = traj(k * ctc.dt)
        return {"q": q_tab, "v": v_tab, "a": a_tab, "k": 0}

    def _plan_cartesian(self, q0, p0_w, p1_w, r1, T):
        wp0 = np.append(p0_w, q0[3])
        wp1 = np.append(p1_w, r1)
        traj = CubicTrajectory([wp0, wp1], [T])
        n = int(round(T / ctc.dt)) + 1
        q_tab = np.zeros((n, 4)); v_tab = np.zeros((n, 4))
        for k in range(n):
            pw, vw, _ = traj(k * ctc.dt)
            th = inverse_kinematics(ctc.world2dh_p(pw[:3], self.base))
            q_tab[k, :3] = ctc.dh2mj_q(th)
            q_tab[k, 3] = pw[3]
            J = jacobian(np.append(th, 0.0))[:, :3]
            v_dh = np.array([-vw[0], -vw[1], vw[2]]) * 1000.0
            v_tab[k, :3] = ctc.swap_dh_mj(np.linalg.solve(J, v_dh))
            v_tab[k, 3] = vw[3]
        a_tab = np.gradient(v_tab, ctc.dt, axis=0)
        return {"q": q_tab, "v": v_tab, "a": a_tab, "k": 0}


    def _publish_state(self, q, v, tau, moving):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(ctc.joint_names)
        js.position = q.tolist()
        js.velocity = v.tolist()
        js.effort = tau.tolist()
        self.pub_js.publish(js)

        p_w = ctc.dh2world_p(forward_kinematics(ctc.mj2dh_q(q))[0], self.base)
        pose = Float64MultiArray()
        pose.data = [p_w[0] * 1000.0, p_w[1] * 1000.0, p_w[2] * 1000.0,
                     float(np.rad2deg(q[3]))]
        self.pub_pose.publish(pose)

        self.pub_moving.publish(Bool(data=moving))


    def sim_loop(self, headless=False):
        viewer = None
        if not headless:
            try:
                viewer = mujoco.viewer.launch_passive(
                    model=self.model, data=self.data,
                    show_left_ui=False, show_right_ui=False)
            except Exception as e:
                self.get_logger().warn(f"뷰어 실행 실패({e}) — 헤드리스로 계속")

        tick = 0
        done_label = None
        try:
            while rclpy.ok() and (viewer is None or viewer.is_running()):
                step_start = time.time()


                q = self.data.qpos[self.qadr].copy()
                v = self.data.qvel[self.dadr].copy()


                started_label = None
                with self.lock:
                    if self.traj is None and self.queue:
                        self.traj = self.queue.popleft()
                        started_label = self.traj["label"]
                    traj = self.traj
                    if traj is not None:
                        k = traj["k"]
                        q_d, v_d, a_d = traj["q"][k], traj["v"][k], traj["a"][k]
                        if k + 1 >= len(traj["q"]):
                            self.hold_q = traj["q"][-1].copy()
                            done_label = traj["label"]
                            self.traj = None
                        else:
                            traj["k"] = k + 1
                        moving = True
                    else:
                        q_d, v_d, a_d = self.hold_q, np.zeros(4), np.zeros(4)
                        moving = False
                if started_label:
                    self.get_logger().info(f"궤적 시작: {started_label}")
                if done_label and not moving:
                    self.get_logger().info(f"궤적 완료: {done_label}")
                    done_label = None


                u = a_d + ctc.Kv * (v_d - v) + ctc.Kp * (q_d - q)
                tau, _ = ctc.ctm_torque(q, v, u, self.I_extra)
                tau = np.clip(tau, self.force_lo, self.force_hi)


                self.data.ctrl[self.act_ids] = tau
                for _ in range(self.n_sub):
                    mujoco.mj_step(self.model, self.data)


                tick += 1
                if tick % STATE_PUB_EVERY == 0:
                    self._publish_state(q, v, tau, moving)
                if viewer is not None:
                    viewer.sync()
                dt_left = ctc.dt - (time.time() - step_start)
                if dt_left > 0:
                    time.sleep(dt_left)
        finally:
            if viewer is not None:
                viewer.close()


def main():
    rclpy.init()
    node = MujocoControllerNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        node.sim_loop(headless="--headless" in sys.argv)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
