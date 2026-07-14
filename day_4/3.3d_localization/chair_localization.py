# -*- coding: utf-8 -*-
import os
import copy
import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
CHAIR_PCD = os.path.join(HERE, "chair.pcd")
ROOM_PCD = os.path.join(HERE, "room.pcd")
VOXEL_SIZE = 0.05
NO_VIS = bool(os.environ.get("LOCALIZE_NO_VIS"))


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, fpfh


def global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def refine_registration(source, target, init_transform, voxel_size):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    return result


def rotation_to_euler_deg(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = 0.0
    return roll, pitch, yaw


def make_dataset():
    demo = o3d.data.DemoCropPointCloud()
    room = o3d.io.read_point_cloud(demo.point_cloud_path)
    vol = o3d.visualization.read_selection_polygon_volume(demo.cropped_json_path)
    chair = vol.crop_point_cloud(room)
    print(f"방 전체: {len(room.points)} pts, 의자 crop: {len(chair.points)} pts")


    T_true = np.eye(4)
    T_true[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz([0.0, 0.7, 0.4])
    T_true[:3, 3] = [1.5, -1.0, 0.5]
    chair.transform(T_true)

    o3d.io.write_point_cloud(CHAIR_PCD, chair)
    o3d.io.write_point_cloud(ROOM_PCD, room)
    print(f"저장 완료: {CHAIR_PCD}\n           {ROOM_PCD}")
    return np.linalg.inv(T_true)


def show_before(chair, room):
    c = copy.deepcopy(chair)
    c.paint_uniform_color([1.0, 0.1, 0.1])
    o3d.visualization.draw_geometries(
        [c, room], window_name="[Before] chair.pcd (red) + room.pcd")


def show_after(chair, room, T):
    c = copy.deepcopy(chair)
    c.paint_uniform_color([1.0, 0.1, 0.1])
    c.transform(T)
    o3d.visualization.draw_geometries(
        [c, room], window_name="[After] FPFH+ICP Localization Result")


def main():

    if os.path.exists(CHAIR_PCD) and os.path.exists(ROOM_PCD):
        print("기존 chair.pcd / room.pcd 사용 (새로 만들려면 두 파일 삭제)")
        T_answer = None
    else:
        T_answer = make_dataset()

    chair = o3d.io.read_point_cloud(CHAIR_PCD)
    room = o3d.io.read_point_cloud(ROOM_PCD)


    if not NO_VIS:
        show_before(chair, room)


    chair_down, chair_fpfh = preprocess_point_cloud(chair, VOXEL_SIZE)
    room_down, room_fpfh = preprocess_point_cloud(room, VOXEL_SIZE)
    result_ransac = global_registration(
        chair_down, room_down, chair_fpfh, room_fpfh, VOXEL_SIZE)
    print("\n[전역 정합 (FPFH+RANSAC)]")
    print(f"  fitness     = {result_ransac.fitness:.4f}")
    print(f"  inlier_rmse = {result_ransac.inlier_rmse:.5f}")


    room_icp = room.voxel_down_sample(VOXEL_SIZE * 0.5)
    room_icp.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 2, max_nn=30))
    result_icp = refine_registration(
        chair, room_icp, result_ransac.transformation, VOXEL_SIZE)
    print("\n[정밀 정합 (ICP point-to-plane)]")
    print(f"  fitness     = {result_icp.fitness:.4f}")
    print(f"  inlier_rmse = {result_icp.inlier_rmse:.5f}")


    T = result_icp.transformation
    t = T[:3, 3]
    roll, pitch, yaw = rotation_to_euler_deg(T[:3, :3])
    np.set_printoptions(precision=4, suppress=True)
    print("\n[Object Pose] chair -> room 변환행렬 T:")
    print(T)
    print(f"\n위치 (x, y, z)          = ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})")
    print(f"자세 (roll, pitch, yaw) = ({roll:.2f}, {pitch:.2f}, {yaw:.2f}) deg")

    if T_answer is not None:
        err_t = np.linalg.norm(T[:3, 3] - T_answer[:3, 3])
        cos = (np.trace(T[:3, :3] @ T_answer[:3, :3].T) - 1) / 2
        err_r = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        print(f"\n[정답 대비 오차] 위치 {err_t*1000:.2f} mm, 회전 {err_r:.3f} deg")


    if not NO_VIS:
        show_after(chair, room, T)


if __name__ == "__main__":
    main()
