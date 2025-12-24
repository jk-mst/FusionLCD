#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings

import sys
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.append('/home/wzb/scancontext')
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "official_sc",
        "/home/wzb/scancontext/SC.py"
    )
    official_sc_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official_sc_module)
    ScanContextProcessor = official_sc_module.ScanContextProcessor
    print("✅ Successfully imported the official SC module")
except Exception as e:
    print(f"❌ Failed to import the official SC module: {e}")
    print("Please check whether the file path is correct")
    raise

warnings.filterwarnings('ignore')


class EnhancedSSCProcessor(ScanContextProcessor):

    def __init__(self, use_forward_filter: bool = True, use_ground_removal: bool = True,
                 ring_num: int = 20, sector_num: int = 60, max_length: float = 80.0,
                 p: float = 0.8, alpha: float = 0.5, min_points_per_cell: int = 1,
                 bev_resolution: float = 1.0, bev_range: float = 30.0,
                 bev_method: str = 'phase_correlation', icp_max_iter: int = 20, icp_tolerance: float = 1e-6):

        super().__init__(use_forward_filter, use_ground_removal, ring_num, sector_num, max_length)

        self.p = p
        self.alpha = alpha
        self.min_points_per_cell = min_points_per_cell

        self.bev_resolution = bev_resolution
        self.bev_range = bev_range
        self.bev_size = int(round(2 * bev_range / bev_resolution))
        self.bev_method = bev_method

        self.icp_max_iter = icp_max_iter
        self.icp_tolerance = icp_tolerance

        print(f"\n🔧 Enhanced sSC parameters:")
        print(f"  sSC: quantile {p*100:.0f}%, fusion weight α={alpha:.1f}")
        print(f"  BEV: resolution {bev_resolution}m, range ±{bev_range}m, grid {self.bev_size}×{self.bev_size}, mode {bev_method}")
        print(f"  ICP: max {icp_max_iter} iterations, tolerance {icp_tolerance}")
        print(f"  Pipeline: sSC coarse match → BEV {bev_method} peak detection → lightweight ICP fine registration")

    def preprocess_pointcloud(self, bin_path: str) -> Tuple[np.ndarray, Dict]:
        print(f"Processing point cloud: {Path(bin_path).name}")
        time_stats = {}

        start_time = time.time()
        points = self.load_pointcloud(bin_path)
        load_time = time.time() - start_time
        time_stats['load'] = load_time
        original_count = len(points)
        print(f"  ✅ Loaded: {original_count} points, time: {load_time:.3f}s")

        if self.use_forward_filter:
            start_time = time.time()
            pre_forward_count = len(points)
            points = self.filter_forward_points(points, forward_angle=90.0)
            forward_filter_time = time.time() - start_time
            time_stats['forward_filter'] = forward_filter_time
            post_forward_count = len(points)
            forward_reduction_pct = (1 - post_forward_count / pre_forward_count) * 100
            print(f"  🎯 Forward filter: {pre_forward_count} → {post_forward_count} points "
                  f"(reduced {forward_reduction_pct:.1f}%), time: {forward_filter_time:.3f}s")
        else:
            time_stats['forward_filter'] = 0.0
            print(f"  ⏭️ Skip forward filter: {len(points)} points")

        if self.use_ground_removal:
            start_time = time.time()
            pre_ground_count = len(points)
            points = self.remove_ground_points(points)
            ground_removal_time = time.time() - start_time
            time_stats['ground_removal'] = ground_removal_time
            post_ground_count = len(points)
            ground_reduction_pct = (1 - post_ground_count / pre_ground_count) * 100
            print(f"  🌱 Ground removal: {pre_ground_count} → {post_ground_count} points "
                  f"(reduced {ground_reduction_pct:.1f}%), time: {ground_removal_time:.3f}s")
        else:
            time_stats['ground_removal'] = 0.0
            print(f"  ⏭️ Skip ground removal: {len(points)} points")

        final_count = len(points)
        total_reduction_pct = (1 - final_count / original_count) * 100 if original_count > 0 else 0
        print(f"  📊 Preprocess done: {original_count} → {final_count} points "
              f"(total reduced {total_reduction_pct:.1f}%)")

        return points, time_stats

    def build_sSC(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        time_stats = {}
        start_time = time.time()

        if len(points) == 0:
            O = np.zeros((self.ring_num, self.sector_num), dtype=np.uint8)
            T = np.zeros((self.ring_num, self.sector_num), dtype=np.float32)
            time_stats['sSC_build'] = time.time() - start_time
            return O, T, time_stats

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        rho = np.sqrt(x*x + y*y)

        valid_mask = rho <= self.max_length
        if not np.any(valid_mask):
            O = np.zeros((self.ring_num, self.sector_num), dtype=np.uint8)
            T = np.zeros((self.ring_num, self.sector_num), dtype=np.float32)
            time_stats['sSC_build'] = time.time() - start_time
            return O, T, time_stats

        rho_valid = rho[valid_mask]
        theta = np.arctan2(y[valid_mask], x[valid_mask]) + np.pi
        z_valid = np.abs(z[valid_mask])

        i_indices = (self.ring_num * rho_valid / self.max_length).astype(np.int32)
        j_indices = (self.sector_num * theta / (2 * np.pi)).astype(np.int32) % self.sector_num

        boundary_mask = (i_indices >= 0) & (i_indices < self.ring_num)
        i_indices = i_indices[boundary_mask]
        j_indices = j_indices[boundary_mask]
        z_valid = z_valid[boundary_mask]

        O = np.zeros((self.ring_num, self.sector_num), dtype=np.uint8)
        T = np.zeros((self.ring_num, self.sector_num), dtype=np.float32)

        if len(i_indices) == 0:
            time_stats['sSC_build'] = time.time() - start_time
            return O, T, time_stats

        grid_indices = i_indices * self.sector_num + j_indices
        unique_grids, inverse_indices = np.unique(grid_indices, return_inverse=True)

        for idx, grid_idx in enumerate(unique_grids):
            cell_mask = inverse_indices == idx
            cell_heights = z_valid[cell_mask]

            if len(cell_heights) >= self.min_points_per_cell:
                i_cell = grid_idx // self.sector_num
                j_cell = grid_idx % self.sector_num

                O[i_cell, j_cell] = 1

                if len(cell_heights) == 1:
                    T[i_cell, j_cell] = cell_heights[0]
                else:
                    rank = int(np.ceil(self.p * len(cell_heights))) - 1
                    rank = max(0, min(rank, len(cell_heights) - 1))
                    T[i_cell, j_cell] = np.partition(cell_heights, rank)[rank]

        build_time = time.time() - start_time
        time_stats['sSC_build'] = build_time

        return O, T, time_stats

    def sSC_match(self, O1: np.ndarray, T1: np.ndarray,
                  O2: np.ndarray, T2: np.ndarray) -> Tuple[float, float, float, int, float, Dict]:
        timing = {}

        iou_start = time.time()
        best_iou = -1.0
        best_delta = 0

        for d in range(self.sector_num):
            Oc_shifted = np.roll(O2, d, axis=1)
            intersection = np.sum(O1 & Oc_shifted)
            union = np.sum(O1 | Oc_shifted)
            iou_val = intersection / union if union > 0 else 0.0

            if iou_val > best_iou:
                best_iou = iou_val
                best_delta = d

        iou_time = time.time() - iou_start
        timing['iou_search'] = iou_time

        cosine_start = time.time()
        Tc_shifted = np.roll(T2, best_delta, axis=1)

        column_similarities = []
        for j in range(self.sector_num):
            col_q = T1[:, j]
            col_c = Tc_shifted[:, j]

            norm_q = np.linalg.norm(col_q)
            norm_c = np.linalg.norm(col_c)

            if norm_q > 1e-6 and norm_c > 1e-6:
                cos_sim = np.dot(col_q, col_c) / (norm_q * norm_c)
                column_similarities.append(cos_sim)

        d_topk = 1.0 - np.mean(column_similarities) if column_similarities else 1.0
        s_topk = 1.0 - d_topk

        cosine_time = time.time() - cosine_start
        timing['cosine_calc'] = cosine_time

        s_occ = best_iou
        s = self.alpha * s_occ + (1 - self.alpha) * s_topk
        best_angle = best_delta * 360.0 / self.sector_num

        return s, s_occ, s_topk, best_delta, best_angle, timing

    def rotate_points(self, points: np.ndarray, yaw_rad: float) -> np.ndarray:
        if len(points) == 0:
            return points

        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)

        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])

        return points @ rotation_matrix.T

    def create_bev_occupancy(self, points: np.ndarray) -> np.ndarray:
        bev_map = np.zeros((self.bev_size, self.bev_size), dtype=np.uint8)

        if len(points) == 0:
            return bev_map

        x, y = points[:, 0], points[:, 1]

        valid_mask = (np.abs(x) <= self.bev_range) & (np.abs(y) <= self.bev_range)
        if not np.any(valid_mask):
            return bev_map

        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        x_idx = ((x_valid + self.bev_range) / self.bev_resolution).astype(np.int32)
        y_idx = ((y_valid + self.bev_range) / self.bev_resolution).astype(np.int32)

        valid_idx_mask = (x_idx >= 0) & (x_idx < self.bev_size) & (y_idx >= 0) & (y_idx < self.bev_size)
        x_idx = x_idx[valid_idx_mask]
        y_idx = y_idx[valid_idx_mask]

        bev_map[y_idx, x_idx] = 1

        return bev_map

    def find_bev_peak_phase_correlation(self, bev1: np.ndarray, bev2: np.ndarray) -> Tuple[float, float, float, Dict]:
        timing = {}
        t0 = time.time()

        img1 = bev1.astype(np.float32)
        img2 = bev2.astype(np.float32)
        assert img1.shape == img2.shape, "BEV size mismatch"

        img1 = img1 - img1.mean()
        img2 = img2 - img2.mean()

        t_fft = time.time()
        F1 = np.fft.rfft2(img1)
        F2 = np.fft.rfft2(img2)
        timing['fft_computation'] = time.time() - t_fft

        t_phase = time.time()
        CPS = F1 * np.conj(F2)
        mag = np.abs(CPS)
        mag[mag < 1e-10] = 1e-10
        C = np.fft.irfft2(CPS / mag).real
        timing['phase_correlation'] = time.time() - t_phase

        t_peak = time.time()
        h, w = C.shape
        py, px = np.unravel_index(np.argmax(C), C.shape)
        C2 = C.copy()
        C2[max(0, py-2):py+3, max(0, px-2):px+3] = 0
        mu, sigma = C2.mean(), C2.std() + 1e-6
        psr = float((C[py, px] - mu) / sigma)
        if px > w//2:
            px -= w
        if py > h//2:
            py -= h
        delta_x = px * self.bev_resolution
        delta_y = py * self.bev_resolution
        confidence = 1.0 / (1.0 + np.exp(-0.5*(psr - 8.0)))
        timing['psr'] = psr
        timing['peak_detection'] = time.time() - t_peak
        timing['bev_peak_detection'] = time.time() - t0
        return delta_x, delta_y, confidence, timing

    def find_bev_peak_fast(self, bev1: np.ndarray, bev2: np.ndarray) -> Tuple[float, float, float, Dict]:
        timing = {}
        start_time = time.time()

        downsample_start = time.time()

        def downsample_2x(img):
            h, w = img.shape
            if h % 2 == 1:
                img = img[:-1, :]
                h -= 1
            if w % 2 == 1:
                img = img[:, :-1]
                w -= 1
            return img.reshape(h//2, 2, w//2, 2).mean(axis=(1, 3))

        bev1_small = downsample_2x(bev1.astype(np.float32))
        bev2_small = downsample_2x(bev2.astype(np.float32))

        timing['downsample'] = time.time() - downsample_start

        search_start = time.time()

        h1, w1 = bev1_small.shape
        h2, w2 = bev2_small.shape

        expected_max_m = 15.0
        max_shift = int(np.ceil(expected_max_m / (2 * self.bev_resolution)))
        max_shift = min(max_shift, h1//2, w1//2)

        best_score = -1
        best_dy, best_dx = 0, 0

        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                y1_start = max(0, dy)
                y1_end = min(h1, h2 + dy)
                x1_start = max(0, dx)
                x1_end = min(w1, w2 + dx)

                y2_start = max(0, -dy)
                y2_end = min(h2, h1 - dy)
                x2_start = max(0, -dx)
                x2_end = min(w2, w1 - dx)

                if y1_end > y1_start and x1_end > x1_start:
                    region1 = bev1_small[y1_start:y1_end, x1_start:x1_end]
                    region2 = bev2_small[y2_start:y2_end, x2_start:x2_end]

                    score = np.sum(region1 * region2)

                    if score > best_score:
                        best_score = score
                        best_dy, best_dx = dy, dx

        timing['window_search'] = time.time() - search_start

        convert_start = time.time()

        delta_x = best_dx * 2 * self.bev_resolution
        delta_y = best_dy * 2 * self.bev_resolution

        y1_start = max(0, best_dy)
        y1_end = min(h1, h2 + best_dy)
        x1_start = max(0, best_dx)
        x1_end = min(w1, w2 + best_dx)

        y2_start = max(0, -best_dy)
        y2_end = min(h2, h1 - best_dy)
        x2_start = max(0, -best_dx)
        x2_end = min(w2, w1 - best_dx)

        if y1_end > y1_start and x1_end > x1_start:
            region1 = bev1_small[y1_start:y1_end, x1_start:x1_end]
            region2 = bev2_small[y2_start:y2_end, x2_start:x2_end]
            r1, r2 = region1 - region1.mean(), region2 - region2.mean()
            ncc = np.sum(r1 * r2) / (np.sqrt(np.sum(r1*r1) * np.sum(r2*r2)) + 1e-6)
            confidence = float(max(0.0, min(1.0, ncc)))
        else:
            confidence = 0.0

        timing['coordinate_conversion'] = time.time() - convert_start
        timing['bev_peak_detection'] = time.time() - start_time

        return delta_x, delta_y, confidence, timing

    def find_bev_peak_original(self, bev1: np.ndarray, bev2: np.ndarray) -> Tuple[float, float, float, Dict]:
        timing = {}
        start_time = time.time()

        correlation = ndimage.correlate(bev1.astype(np.float32), bev2.astype(np.float32), mode='constant')

        peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
        peak_y, peak_x = peak_idx

        center = self.bev_size // 2
        delta_x = (peak_x - center) * self.bev_resolution
        delta_y = (peak_y - center) * self.bev_resolution

        confidence = correlation[peak_y, peak_x] / (np.sum(bev1) * np.sum(bev2) + 1e-6)

        timing['bev_peak_detection'] = time.time() - start_time

        return delta_x, delta_y, confidence, timing

    def lightweight_icp(self, source_points: np.ndarray, target_points: np.ndarray,
                        initial_transform: np.ndarray,
                        multiscale_voxels=(1.0, 0.5, 0.25),
                        k_normals=20,
                        trim_ratio=0.8,
                        robust='huber',
                        robust_delta=0.5,
                        max_iter_per_level=20) -> Tuple[np.ndarray, float, List[float], Dict]:
        t_all = time.time()
        if len(source_points) == 0 or len(target_points) == 0:
            timing = {'icp_total': 0.0, 'global_rmse': float('inf'), 'iterations': 0}
            return initial_transform, float('inf'), [], timing

        np.random.seed(42)

        def voxel_downsample(P, voxel):
            if len(P) == 0 or voxel <= 0:
                return P
            keys = np.floor(P[:, :2] / voxel).astype(np.int64)
            _, idx = np.unique(keys, axis=0, return_index=True)
            return P[idx]

        T = initial_transform.astype(np.float32).copy()
        error_history = []
        total_iters = 0

        for lvl, vox in enumerate(multiscale_voxels):
            src = voxel_downsample(source_points, vox)
            tgt = voxel_downsample(target_points, vox)

            tree = cKDTree(tgt[:, :2])

            tgt_normals = self._estimate_normals(tgt, k=k_normals)

            base_radius = 1.5 if lvl == 0 else (1.0 if lvl == 1 else 0.6)
            radius = base_radius

            for it in range(max_iter_per_level):
                total_iters += 1
                src_T = self.apply_transform(src, T)

                dist, nn = tree.query(src_T[:, :2], k=1, distance_upper_bound=radius)
                msk = np.isfinite(dist)
                if msk.sum() < 20:
                    if radius < base_radius * 2.0:
                        radius *= 1.5
                        continue
                    else:
                        break

                A = src_T[msk]
                B = tgt[nn[msk]]
                N = tgt_normals[nn[msk]]

                r = np.sum(N * (B - A), axis=1)

                if 0.5 <= trim_ratio < 1.0 and len(r) > 50:
                    k = int(len(r) * trim_ratio)
                    idx = np.argpartition(np.abs(r), k)[:k]
                    A, B, N, r = A[idx], B[idx], N[idx], r[idx]

                w = self._robust_weight(r, delta=robust_delta, kind=robust).reshape(-1, 1)

                J = np.zeros((len(A), 3), dtype=np.float32)
                J[:, 0] = N[:, 0]
                J[:, 1] = N[:, 1]
                J[:, 2] = N[:, 0] * (-A[:, 1]) + N[:, 1] * (A[:, 0])

                W = w.squeeze()
                JW = J * W[:, None]
                H = JW.T @ J + 1e-6 * np.eye(3, dtype=np.float32)
                b = JW.T @ r

                try:
                    d = np.linalg.solve(H, b)
                except np.linalg.LinAlgError:
                    break

                dyaw = float(d[2])
                c, s = np.cos(dyaw), np.sin(dyaw)
                T_inc = np.eye(4, dtype=np.float32)
                T_inc[0, 0] = c
                T_inc[0, 1] = -s
                T_inc[1, 0] = s
                T_inc[1, 1] = c
                T_inc[0, 3] = float(d[0])
                T_inc[1, 3] = float(d[1])
                T = T_inc @ T

                rmse_plane = float(np.sqrt(np.average(r*r, weights=W) + 1e-12))
                if abs(d[0]) < 1e-4 and abs(d[1]) < 1e-4 and abs(dyaw) < np.deg2rad(0.05):
                    error_history.append(rmse_plane)
                    break

                if it == max_iter_per_level - 1:
                    error_history.append(rmse_plane)

        tree_full = cKDTree(target_points[:, :2])
        src_final = self.apply_transform(source_points, T)[:, :2]
        global_dist, _ = tree_full.query(src_final, k=1)
        global_rmse = float(np.sqrt(np.mean(global_dist**2)))

        timing = {
            'icp_total': time.time() - t_all,
            'global_rmse': global_rmse,
            'iterations': total_iters,
            'rmse2d': np.nan,
            'inlier_ratio': np.nan,
            'global_inlier_ratio': float(np.mean(global_dist < 0.5)),
        }
        final_error = error_history[-1] if error_history else float('inf')
        return T, final_error, error_history, timing

    def apply_transform(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points

        R, t = transform[:3, :3], transform[:3, 3]
        return (points @ R.T) + t

    def _estimate_normals(self, P: np.ndarray, k: int = 20) -> np.ndarray:
        if len(P) == 0:
            return np.zeros_like(P)
        tree = cKDTree(P[:, :2])
        normals = np.zeros((len(P), 3), dtype=np.float32)
        for i, p in enumerate(P):
            _, idx = tree.query(p[:2], k=min(k, len(P)))
            N = P[idx, :2]
            mu = N.mean(axis=0)
            X = N - mu
            C = X.T @ X / max(len(N)-1, 1)
            eigvals, eigvecs = np.linalg.eigh(C)
            n2 = eigvecs[:, 0]
            n = np.array([n2[0], n2[1], 0.0], dtype=np.float32)
            if np.dot(n[:2], (mu - p[:2])) < 0:
                n = -n
            normals[i] = n / (np.linalg.norm(n) + 1e-12)
        return normals

    def _robust_weight(self, r: np.ndarray, delta: float = 0.5, kind: str = 'huber') -> np.ndarray:
        a = np.abs(r)
        if kind == 'huber':
            w = np.where(a <= delta, 1.0, delta / (a + 1e-12))
        elif kind == 'tukey':
            c = delta
            w = np.where(a < c, (1 - (a/c)**2)**2, 0.0)
        else:
            w = np.ones_like(r)
        return w.astype(np.float32)

    def create_initial_transform(self, delta_x: float, delta_y: float, delta_yaw: float) -> np.ndarray:
        c, s = np.cos(delta_yaw), np.sin(delta_yaw)
        R = np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]], dtype=np.float32)

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3,  3] = np.array([delta_x, delta_y, 0], dtype=np.float32)
        return T

    def process_enhanced_matching(self, frame1_path: str, frame2_path: str) -> Dict:
        total_start = time.time()

        print("=" * 80)
        print("🚀 Enhanced sSC multi-stage matching: sSC coarse match → BEV peak detection → lightweight ICP fine registration")
        print("=" * 80)

        print(f"\n🎯 Stage 1: sSC coarse matching")
        print(f"📁 Processing frame 1: {Path(frame1_path).name}")

        frame1_prep_start = time.time()
        P1, prep_time1_dict = self.preprocess_pointcloud(frame1_path)
        frame1_prep_time = time.time() - frame1_prep_start

        frame1_build_start = time.time()
        O1, T1, build_time1_dict = self.build_sSC(P1)
        frame1_build_time = time.time() - frame1_build_start

        print(f"📁 Processing frame 2: {Path(frame2_path).name}")

        frame2_prep_start = time.time()
        P2, prep_time2_dict = self.preprocess_pointcloud(frame2_path)
        frame2_prep_time = time.time() - frame2_prep_start

        frame2_build_start = time.time()
        O2, T2, build_time2_dict = self.build_sSC(P2)
        frame2_build_time = time.time() - frame2_build_start

        print(f"🔍 Running sSC two-stage matching...")
        match_start = time.time()
        similarity, s_occ, s_topk, best_delta, best_angle, match_timing = self.sSC_match(O1, T1, O2, T2)
        similarity_calc_time = time.time() - match_start

        print(f"  ✅ sSC done: similarity={similarity:.4f}, yaw={best_angle:.1f}°")

        print(f"\n🎯 Stage 2: BEV peak detection")

        delta_yaw_rad = np.radians(best_angle)
        print(f"🔄 Rotating frame 1 by {-best_angle:.1f}° to align with frame 2 (from sSC estimate)")

        bev_start = time.time()
        P1_rotated = self.rotate_points(P1, -delta_yaw_rad)

        print(f"📊 Building BEV occupancy maps (resolution {self.bev_resolution}m)...")
        bev1 = self.create_bev_occupancy(P1_rotated)
        bev2 = self.create_bev_occupancy(P2)

        print(f"🔍 BEV {self.bev_method} peak detection...")
        if self.bev_method == 'phase_correlation':
            delta_x, delta_y, bev_confidence, bev_timing = self.find_bev_peak_phase_correlation(bev2, bev1)
        elif self.bev_method == 'fast':
            delta_x, delta_y, bev_confidence, bev_timing = self.find_bev_peak_fast(bev2, bev1)
        elif self.bev_method == 'optimized':
            delta_x, delta_y, bev_confidence, bev_timing = self.find_bev_peak_fast(bev2, bev1)
        else:
            delta_x, delta_y, bev_confidence, bev_timing = self.find_bev_peak_original(bev2, bev1)
        bev_total_time = time.time() - bev_start

        if self.bev_method == 'phase_correlation':
            psr = bev_timing.get('psr', None)
            print(f"  ✅ Phase correlation done: offset=({delta_x:.2f}m, {delta_y:.2f}m), "
                  f"PSR={psr:.2f}, confidence={bev_confidence:.2f}, "
                  f"time={bev_timing['bev_peak_detection']*1000:.2f}ms")
        else:
            print(f"  ✅ BEV done: offset=({delta_x:.2f}m, {delta_y:.2f}m), confidence={bev_confidence:.4f}")

        print(f"\n🎯 Stage 3: Lightweight ICP fine registration")

        initial_transform = self.create_initial_transform(delta_x, delta_y, -delta_yaw_rad)
        print(f"🎯 ICP initial guess (Frame1→Frame2): Δx={delta_x:.2f}m, Δy={delta_y:.2f}m, Δyaw={-best_angle:.1f}°")

        original_icp_max_iter = self.icp_max_iter
        psr = bev_timing.get('psr', 0.0)
        confidence_ok = (psr > 8.0) or (bev_confidence >= 0.75)
        if s_occ >= 0.6 and confidence_ok:
            self.icp_max_iter = min(self.icp_max_iter, 5)
            print(f"  💪 High-quality match detected (s_occ={s_occ:.3f}, PSR={psr:.1f}, BEV confidence={bev_confidence:.3f}); reducing ICP iterations to {self.icp_max_iter}")

        icp_start = time.time()
        final_transform, final_error, error_history, icp_timing = self.lightweight_icp(P1, P2, initial_transform)

        self.icp_max_iter = original_icp_max_iter
        icp_total_time = time.time() - icp_start

        final_translation = final_transform[:3, 3]
        final_rotation_matrix = final_transform[:3, :3]
        final_yaw = np.arctan2(final_rotation_matrix[1, 0], final_rotation_matrix[0, 0])

        rmse2d = icp_timing.get('rmse2d', None)
        inlier_ratio = icp_timing.get('inlier_ratio', None)
        global_rmse = icp_timing.get('global_rmse', None)
        global_inlier_ratio = icp_timing.get('global_inlier_ratio', None)

        rmse2d_str = f"{rmse2d:.3f}m" if rmse2d is not None else "N/A"
        inlier_ratio_str = f"{inlier_ratio:.2f}" if inlier_ratio is not None else "N/A"
        global_rmse_str = f"{global_rmse:.3f}m" if global_rmse is not None else "N/A"
        global_inlier_str = f"{global_inlier_ratio:.2f}" if global_inlier_ratio is not None else "N/A"

        print(f"  ✅ ICP done: {len(error_history)} iterations, "
              f"final error (last iteration)={final_error:.4f}m")
        print(f"     Local quality: RMSE2D={rmse2d_str}, inlier ratio={inlier_ratio_str}")
        print(f"     Global quality: RMSE={global_rmse_str}, global inlier ratio={global_inlier_str}")
        print(f"     Final transform: translation=({final_translation[0]:.3f}, {final_translation[1]:.3f}), yaw={np.degrees(final_yaw):.2f}°")

        total_time = time.time() - total_start

        stage1_time = frame1_prep_time + frame1_build_time + frame2_prep_time + frame2_build_time + similarity_calc_time
        stage2_time = bev_total_time
        stage3_time = icp_total_time
        calculated_total = stage1_time + stage2_time + stage3_time

        print(f"\n⚡ Multi-stage matching performance:")
        print(f"  🎯 Stage 1 - sSC (incl. preprocessing): {stage1_time*1000:.1f}ms")
        print(f"     • Preprocessing: {(frame1_prep_time + frame2_prep_time)*1000:.1f}ms")
        print(f"     • sSC build: {(frame1_build_time + frame2_build_time)*1000:.1f}ms")
        print(f"     • IoU prefilter: {match_timing['iou_search']*1000000:.0f}μs")
        print(f"     • Cosine calc: {match_timing['cosine_calc']*1000:.1f}ms")
        print(f"  🎯 Stage 2 - BEV: {stage2_time*1000:.1f}ms")
        print(f"     • Peak detection: {bev_timing['bev_peak_detection']*1000:.1f}ms")
        print(f"  🎯 Stage 3 - ICP: {stage3_time*1000:.1f}ms")
        print(f"     • {len(error_history)} iterations")
        print(f"  🏁 Total: {calculated_total*1000:.1f}ms (actual: {total_time*1000:.1f}ms)")

        return {
            'sSC_similarity': similarity,
            'sSC_occupancy_similarity': s_occ,
            'sSC_feature_similarity': s_topk,
            'sSC_yaw_degrees': best_angle,
            'sSC_yaw_radians': delta_yaw_rad,

            'BEV_delta_x': delta_x,
            'BEV_delta_y': delta_y,
            'BEV_confidence': bev_confidence,
            'BEV_psr': bev_timing.get('psr', None),
            'BEV_ref_P2': bev2,
            'BEV_query_rot_P1': bev1,

            'ICP_final_transform': final_transform,
            'ICP_final_error': final_error,
            'ICP_error_history': error_history,
            'ICP_iterations': len(error_history),
            'ICP_final_translation': final_translation,
            'ICP_final_yaw_degrees': np.degrees(final_yaw),
            'ICP_rmse2d': icp_timing.get('rmse2d', None),
            'ICP_inlier_ratio': icp_timing.get('inlier_ratio', None),
            'ICP_global_rmse': icp_timing.get('global_rmse', None),
            'ICP_global_inlier_ratio': icp_timing.get('global_inlier_ratio', None),

            'time_sSC_ms': stage1_time * 1000,
            'time_BEV_ms': stage2_time * 1000,
            'time_ICP_ms': stage3_time * 1000,
            'time_total_ms': calculated_total * 1000,
            'time_total_actual_ms': total_time * 1000,

            'frame1_prep_time_ms': frame1_prep_time * 1000,
            'frame1_build_time_ms': frame1_build_time * 1000,
            'frame2_prep_time_ms': frame2_prep_time * 1000,
            'frame2_build_time_ms': frame2_build_time * 1000,
            'iou_time_us': match_timing['iou_search'] * 1000000,
            'cosine_time_ms': match_timing['cosine_calc'] * 1000,
            'bev_peak_time_ms': bev_timing['bev_peak_detection'] * 1000,
            'icp_time_ms': icp_timing['icp_total'] * 1000,

            'points1': P1,
            'points2': P2,
            'points1_rotated': P1_rotated,
            'points1_count': len(P1),
            'points2_count': len(P2),

            'occupancy1': O1,
            'occupancy2': O2,
            'features1': T1,
            'features2': T2,

            'config': {
                'sSC_quantile_p': self.p,
                'sSC_fusion_alpha': self.alpha,
                'BEV_resolution': self.bev_resolution,
                'BEV_range': self.bev_range,
                'ICP_max_iter': self.icp_max_iter,
                'ICP_tolerance': self.icp_tolerance
            }
        }

    def visualize_enhanced_results(self, results: Dict, save_path: Optional[str] = None):
        fig = plt.figure(figsize=(24, 16))

        gs = fig.add_gridspec(5, 8, hspace=0.3, wspace=0.3)

        ax_ssc = [fig.add_subplot(gs[0, i]) for i in range(3)]
        ax_polar = [fig.add_subplot(gs[0, i+3], projection='polar') for i in range(2)]

        O1, O2 = results['occupancy1'], results['occupancy2']

        x_edges = np.arange(self.sector_num + 1) - 0.5
        y_edges = np.arange(self.ring_num + 1) - 0.5
        ax_ssc[0].pcolormesh(x_edges, y_edges, O1, cmap='binary', shading='flat', edgecolors='gray', linewidth=0.01)
        ax_ssc[0].set_title('Frame 1 sSC Occupancy Map', fontweight='bold')
        ax_ssc[0].set_xlabel('Sector')
        ax_ssc[0].set_ylabel('Ring')
        ax_ssc[0].set_xticks(np.arange(0, self.sector_num, 10))
        ax_ssc[0].set_yticks(np.arange(0, self.ring_num, 5))
        ax_ssc[0].set_aspect('auto')
        ax_ssc[0].set_xlim(-0.5, self.sector_num - 0.5)
        ax_ssc[0].set_ylim(-0.5, self.ring_num - 0.5)
        ax_ssc[0].invert_yaxis()

        x_edges = np.arange(self.sector_num + 1) - 0.5
        y_edges = np.arange(self.ring_num + 1) - 0.5
        ax_ssc[1].pcolormesh(x_edges, y_edges, O2, cmap='binary', shading='flat', edgecolors='gray', linewidth=0.5)
        ax_ssc[1].set_title('Frame 2 sSC Occupancy Map', fontweight='bold')
        ax_ssc[1].set_xlabel('Sector')
        ax_ssc[1].set_ylabel('Ring')
        ax_ssc[1].set_xticks(np.arange(0, self.sector_num, 10))
        ax_ssc[1].set_yticks(np.arange(0, self.ring_num, 5))
        ax_ssc[1].set_aspect('auto')
        ax_ssc[1].set_xlim(-0.5, self.sector_num - 0.5)
        ax_ssc[1].set_ylim(-0.5, self.ring_num - 0.5)
        ax_ssc[1].invert_yaxis()

        T1, T2 = results['features1'], results['features2']
        im_t = ax_ssc[2].imshow(T1, cmap='viridis', aspect='auto')
        ax_ssc[2].set_title('Frame 1 sSC Quantile Features', fontweight='bold')
        ax_ssc[2].set_xlabel('Sector')
        ax_ssc[2].set_ylabel('Ring')
        ax_ssc[2].set_xticks(np.arange(0, self.sector_num, 10))
        ax_ssc[2].set_yticks(np.arange(0, self.ring_num, 5))
        ax_ssc[2].set_yticklabels(np.arange(0, self.ring_num, 5))
        ax_ssc[2].invert_yaxis()
        plt.colorbar(im_t, ax=ax_ssc[2], label='Quantile Height')

        P1, P2 = results['points1'], results['points2']

        if len(P1) > 0:
            x1, y1 = P1[:, 0], P1[:, 1]
            rho1 = np.sqrt(x1**2 + y1**2)
            theta1 = np.arctan2(y1, x1)

            valid_mask1 = rho1 <= self.max_length
            rho1_valid = rho1[valid_mask1]
            theta1_valid = theta1[valid_mask1]

            if len(rho1_valid) > 5000:
                sample_idx = np.random.choice(len(rho1_valid), 5000, replace=False)
                rho1_valid = rho1_valid[sample_idx]
                theta1_valid = theta1_valid[sample_idx]

            ax_polar[0].scatter(theta1_valid, rho1_valid, c='red', s=1, alpha=0.6)
            ax_polar[0].set_title('Frame 1 Polar Projection', fontweight='bold', pad=20)
            ax_polar[0].set_ylim(0, self.max_length)
            ax_polar[0].set_rticks([20, 40, 60, 80])
            ax_polar[0].grid(True)

        if len(P2) > 0:
            x2, y2 = P2[:, 0], P2[:, 1]
            rho2 = np.sqrt(x2**2 + y2**2)
            theta2 = np.arctan2(y2, x2)

            valid_mask2 = rho2 <= self.max_length
            rho2_valid = rho2[valid_mask2]
            theta2_valid = theta2[valid_mask2]

            if len(rho2_valid) > 5000:
                sample_idx = np.random.choice(len(rho2_valid), 5000, replace=False)
                rho2_valid = rho2_valid[sample_idx]
                theta2_valid = theta2_valid[sample_idx]

            ax_polar[1].scatter(theta2_valid, rho2_valid, c='blue', s=1, alpha=0.6)
            ax_polar[1].set_title('Frame 2 Polar Projection', fontweight='bold', pad=20)
            ax_polar[1].set_ylim(0, self.max_length)
            ax_polar[1].set_rticks([20, 40, 60, 80])
            ax_polar[1].grid(True)

        ax_bev = [fig.add_subplot(gs[1, i*2:(i+1)*2]) for i in range(3)]

        bev_p2 = results['BEV_ref_P2']
        bev_p1_rot = results['BEV_query_rot_P1']

        ax_bev[0].imshow(bev_p2, cmap='Blues', origin='lower', extent=[-self.bev_range, self.bev_range, -self.bev_range, self.bev_range])
        ax_bev[0].set_title('Frame 2 BEV (Reference)', fontweight='bold')
        ax_bev[0].set_xlabel('X (m)')
        ax_bev[0].set_ylabel('Y (m)')
        ax_bev[0].grid(True, alpha=0.3)

        ax_bev[1].imshow(bev_p1_rot, cmap='Reds', origin='lower', extent=[-self.bev_range, self.bev_range, -self.bev_range, self.bev_range])
        ax_bev[1].set_title('Frame 1 BEV (After sSC Rotation)', fontweight='bold')
        ax_bev[1].set_xlabel('X (m)')
        ax_bev[1].set_ylabel('Y (m)')
        ax_bev[1].grid(True, alpha=0.3)

        bev_overlap = np.zeros((self.bev_size, self.bev_size, 3))
        bev_overlap[:, :, 2] = bev_p2
        bev_overlap[:, :, 0] = bev_p1_rot
        bev_overlap[:, :, 1] = bev_p2 & bev_p1_rot

        ax_bev[2].imshow(bev_overlap, origin='lower', extent=[-self.bev_range, self.bev_range, -self.bev_range, self.bev_range])
        ax_bev[2].set_title('BEV Overlap Analysis', fontweight='bold')
        ax_bev[2].set_xlabel('X (m)')
        ax_bev[2].set_ylabel('Y (m)')

        ax_bev[2].plot(results['BEV_delta_x'], results['BEV_delta_y'], 'yo', markersize=10, label=f"Peak ({results['BEV_delta_x']:.2f}, {results['BEV_delta_y']:.2f})")
        ax_bev[2].legend()
        ax_bev[2].grid(True, alpha=0.3)

        ax_icp = fig.add_subplot(gs[2, :4])

        error_history = results['ICP_error_history']
        if error_history:
            ax_icp.plot(error_history, 'bo-', linewidth=2, markersize=6)
            ax_icp.set_xlabel('ICP Iteration Number')
            ax_icp.set_ylabel('Average Registration Error (m)')
            ax_icp.set_title(f'ICP Convergence Process ({len(error_history)} iterations)', fontweight='bold')
            ax_icp.grid(True, alpha=0.5)
            ax_icp.text(0.7, 0.8, f'Final Error: {results["ICP_final_error"]:.4f}m',
                        transform=ax_icp.transAxes, fontsize=12,
                        bbox=dict(boxstyle="round", facecolor='lightblue'))

        ax_pc = [fig.add_subplot(gs[3, i*2:(i+1)*2]) for i in range(3)]

        P1, P2 = results['points1'], results['points2']
        P1_rotated = results['points1_rotated']
        delta_x = results['BEV_delta_x']
        delta_y = results['BEV_delta_y']

        all_points = []
        if len(P2) > 0:
            all_points.append(P2)
        if len(P1_rotated) > 0:
            P1_ssc_bev = P1_rotated.copy()
            P1_ssc_bev[:, 0] += delta_x
            P1_ssc_bev[:, 1] += delta_y
            all_points.append(P1_ssc_bev)

        if all_points:
            all_coords = np.vstack(all_points)
            x_min, x_max = np.min(all_coords[:, 0]), np.max(all_coords[:, 0])
            y_min, y_max = np.min(all_coords[:, 1]), np.max(all_coords[:, 1])
            x_range = x_max - x_min
            y_range = y_max - y_min
            margin = 0.1
            x_min -= x_range * margin
            x_max += x_range * margin
            y_min -= y_range * margin
            y_max += y_range * margin
        else:
            x_min, x_max, y_min, y_max = -50, 50, -50, 50

        if len(P1) > 0:
            ax_pc[0].scatter(P1[:, 0], P1[:, 1], c='red', s=1, alpha=0.6, label='Frame 1')
        if len(P2) > 0:
            ax_pc[0].scatter(P2[:, 0], P2[:, 1], c='blue', s=1, alpha=0.6, label='Frame 2')
        ax_pc[0].set_title('Original Point Cloud Comparison', fontweight='bold')
        ax_pc[0].set_xlabel('X (m)')
        ax_pc[0].set_ylabel('Y (m)')
        ax_pc[0].legend()
        ax_pc[0].grid(True, alpha=0.3)
        ax_pc[0].axis('equal')
        ax_pc[0].set_xlim(x_min, x_max)
        ax_pc[0].set_ylim(y_min, y_max)

        if len(P2) > 0:
            ax_pc[1].scatter(P2[:, 0], P2[:, 1], c='blue', s=1, alpha=0.6, label='Frame 2')
        if len(P1_rotated) > 0:
            P1_ssc_bev = P1_rotated.copy()
            P1_ssc_bev[:, 0] += delta_x
            P1_ssc_bev[:, 1] += delta_y
            ax_pc[1].scatter(P1_ssc_bev[:, 0], P1_ssc_bev[:, 1], c='red', s=1, alpha=0.6, label='Frame 1 (sSC+BEV Aligned)')
        ax_pc[1].set_title(f'sSC+BEV Alignment Result (Δx={delta_x:.2f}m, Δy={delta_y:.2f}m)', fontweight='bold')
        ax_pc[1].set_xlabel('X (m)')
        ax_pc[1].set_ylabel('Y (m)')
        ax_pc[1].legend(loc='upper right')
        ax_pc[1].grid(True, alpha=0.3)
        ax_pc[1].axis('equal')
        ax_pc[1].set_xlim(x_min, x_max)
        ax_pc[1].set_ylim(y_min, y_max)

        if len(P2) > 0:
            ax_pc[2].scatter(P2[:, 0], P2[:, 1], c='blue', s=1, alpha=0.6, label='Frame 2')
        if len(P1) > 0:
            P1_final = self.apply_transform(P1, results['ICP_final_transform'])
            ax_pc[2].scatter(P1_final[:, 0], P1_final[:, 1], c='red', s=1, alpha=0.6, label='Frame 1 (ICP Fine Aligned)')
        ax_pc[2].set_title('ICP Final Alignment Result', fontweight='bold')
        ax_pc[2].set_xlabel('X (m)')
        ax_pc[2].set_ylabel('Y (m)')
        ax_pc[2].legend()
        ax_pc[2].grid(True, alpha=0.3)
        ax_pc[2].axis('equal')
        ax_pc[2].set_xlim(x_min, x_max)
        ax_pc[2].set_ylim(y_min, y_max)

        fig.suptitle('Enhanced sSC Multi-stage Matching Complete Analysis Results', fontsize=20, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📸 Visualization saved: {save_path}")

        plt.show()


def main():
    base_path = "/home/wzb/0000000000alldatas/dataset/sequences/00/velodyne"
    frame1_path = f"{base_path}/001580.bin"
    frame2_path = f"{base_path}/000134.bin"

    enhanced_processor = EnhancedSSCProcessor(
        use_forward_filter=True,
        use_ground_removal=True,
        ring_num=20,
        sector_num=60,
        max_length=80.0,
        p=0.8,
        alpha=0.5,
        min_points_per_cell=1,
        bev_resolution=0.5,
        bev_range=25.0,
        bev_method='phase_correlation',
        icp_max_iter=30,
        icp_tolerance=1e-3
    )

    try:
        print("🎯 Testing enhanced sSC multi-stage matching")
        results = enhanced_processor.process_enhanced_matching(frame1_path, frame2_path)

        save_path = "/home/wzb/zzzzzzzzzzzz生成论文图/result_1580_---134.png"
        enhanced_processor.visualize_enhanced_results(results, save_path)

        return results

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please check whether the point cloud file path is correct")
        return None
    except Exception as e:
        print(f"❌ An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("🚀 Enhanced sSC: sSC coarse match → BEV peak detection → lightweight ICP fine registration")
    print("Purpose: multi-stage point cloud matching from coarse to fine, balancing speed and accuracy")
    print("-" * 80)

    results = main()

    if results:
        print("\n✅ Enhanced sSC test completed!")
        print(f"Total processing time: {results['time_total_ms']:.1f}ms")
        print(f"sSC similarity: {results['sSC_similarity']:.4f}")
        print(f"BEV translation estimate: ({results['BEV_delta_x']:.2f}, {results['BEV_delta_y']:.2f})m")
        print(f"ICP final error: {results['ICP_final_error']:.4f}m")
        print("🎯 Multi-stage matching successfully achieved efficient coarse-to-fine point cloud registration!")
