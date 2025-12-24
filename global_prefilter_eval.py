import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import faiss
import warnings
import time

warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')


class SuperPointNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, 3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, 3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, 3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, 3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, 3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, 3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, 3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, 3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, 3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, 1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, 3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, 256, 1, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))

        semi = self.convPb(F.relu(self.convPa(x)))
        desc = self.convDb(F.relu(self.convDa(x)))
        dn = F.normalize(desc, p=2, dim=1)
        return semi, dn


class AdaptiveGeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        gem_pool = F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)
        max_pool = F.adaptive_max_pool2d(x, (1, 1))
        return torch.cat([gem_pool, max_pool], dim=1)


class SuperPointGeM:
    def __init__(self, weights_path, device='cuda', gem_p=3):
        self.device = device
        self.net = SuperPointNet({}).to(device).eval()

        try:
            checkpoint = torch.load(weights_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                self.net.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.net.load_state_dict(checkpoint['state_dict'])
            else:
                self.net.load_state_dict(checkpoint)
            print(f"Successfully loaded SuperPoint weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise

        self.gem_pool = AdaptiveGeM(p=gem_p).to(device)

    def extract_global_descriptor(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = img.shape
        h_new = (h // 8) * 8
        w_new = (w // 8) * 8
        img = cv2.resize(img, (w_new, h_new))

        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, desc = self.net(img_tensor)

            descriptors = []

            gem_desc = self.gem_pool(desc)
            gem_desc = F.normalize(gem_desc.squeeze(), p=2, dim=0)
            descriptors.append(gem_desc.cpu().numpy())

            spatial_desc = self.spatial_pyramid_pooling(desc)
            descriptors.append(spatial_desc.cpu().numpy())

            regional_desc = self.regional_pooling(desc)
            descriptors.append(regional_desc.cpu().numpy())

            global_desc = np.concatenate(descriptors)
            global_desc = global_desc / (np.linalg.norm(global_desc) + 1e-6)

        return global_desc

    def spatial_pyramid_pooling(self, x, levels=(1, 2, 4)):
        _, _, H, W = x.shape
        pooled = []
        for level in levels:
            h_step = H // level
            w_step = W // level
            for i in range(level):
                for j in range(level):
                    h_start = i * h_step
                    h_end = (i + 1) * h_step if i < level - 1 else H
                    w_start = j * w_step
                    w_end = (j + 1) * w_step if j < level - 1 else W
                    region = x[:, :, h_start:h_end, w_start:w_end]
                    pooled.append(F.adaptive_avg_pool2d(region, (1, 1)))
        spp = torch.cat(pooled, dim=1)
        return F.normalize(spp.squeeze(), p=2, dim=0)

    def regional_pooling(self, x):
        _, _, H, W = x.shape
        regions = []
        region_coords = [
            (0, H // 2, 0, W // 2),
            (0, H // 2, W // 4, 3 * W // 4),
            (0, H // 2, W // 2, W),
            (H // 4, 3 * H // 4, 0, W // 2),
            (H // 4, 3 * H // 4, W // 2, W),
            (H // 2, H, 0, W // 2),
            (H // 2, H, W // 4, 3 * W // 4),
            (H // 2, H, W // 2, W),
        ]
        for h1, h2, w1, w2 in region_coords:
            region = x[:, :, h1:h2, w1:w2]
            regions.append(F.adaptive_avg_pool2d(region, (1, 1)))
        reg = torch.cat(regions, dim=1)
        return F.normalize(reg.squeeze(), p=2, dim=0)


class RealtimeLoopClosureDetector:
    def __init__(self, superpoint_path, similarity_threshold=0.3,
                 min_time_diff=100, device='cuda', debug=False,
                 check_interval=5):
        self.similarity_threshold = similarity_threshold
        self.min_time_diff = min_time_diff
        self.device = device
        self.debug = debug
        self.check_interval = check_interval

        self.feature_extractor = SuperPointGeM(superpoint_path, device=device)

        self.loop_closures = []
        self.similarity_scores = []

        self.frame_count = 0
        self.encoding_times = []
        self.matching_times = []

        self.faiss_index = None
        self.faiss_ids = []

    def init_faiss_index(self, descriptor_dim):
        self.faiss_index = faiss.IndexFlatL2(descriptor_dim)
        if self.debug:
            print(f"Initialized FAISS IndexFlatL2 dim={descriptor_dim}")

    def encode_frame(self, img, frame_idx):
        t0 = time.time()

        global_desc = self.feature_extractor.extract_global_descriptor(img)

        if self.faiss_index is None:
            self.init_faiss_index(len(global_desc))

        self.faiss_index.add(global_desc.reshape(1, -1).astype(np.float32))
        self.faiss_ids.append(frame_idx)

        dt = time.time() - t0
        self.encoding_times.append(dt)

        if self.debug:
            print(f"Frame {frame_idx}: Encoded {dt:.4f}s, FAISS size={self.faiss_index.ntotal}")

        return global_desc

    def check_loop_closure(self, query_desc, query_idx):
        t0 = time.time()

        if len(self.faiss_ids) < self.min_time_diff:
            if self.debug:
                print(f"Frame {query_idx}: Not enough history ({len(self.faiss_ids)} < {self.min_time_diff})")
            return []

        k = min(len(self.faiss_ids), 100)
        if k == 0:
            if self.debug:
                print(f"Frame {query_idx}: Empty FAISS index")
            return []

        distances, indices = self.faiss_index.search(query_desc.reshape(1, -1).astype(np.float32), k)
        distances = distances[0]
        similarities = np.exp(-distances * 2)
        indices = indices[0]

        valid_pairs = []
        valid_sims_all = []

        for faiss_pos, sim in zip(indices, similarities):
            if faiss_pos < len(self.faiss_ids):
                stored_frame_idx = self.faiss_ids[faiss_pos]
                if abs(query_idx - stored_frame_idx) >= self.min_time_diff:
                    valid_sims_all.append(sim)
                    if sim > self.similarity_threshold:
                        valid_pairs.append((stored_frame_idx, sim))

        if not valid_sims_all:
            if self.debug:
                print(f"Frame {query_idx}: No valid candidates after time filtering")
            return []

        self.similarity_scores.extend(valid_sims_all)

        valid_pairs.sort(key=lambda x: x[1], reverse=True)
        top_k = valid_pairs[:5]

        dt = time.time() - t0
        self.matching_times.append(dt)

        if self.debug:
            if top_k:
                print(f"Frame {query_idx}: Loop! best={top_k[0][0]} score={top_k[0][1]:.3f} time={dt:.4f}s")
            else:
                print(f"Frame {query_idx}: No loop (max={max(valid_sims_all):.3f}) time={dt:.4f}s")

        return top_k

    def process_frame(self, img, frame_idx):
        query_desc = self.encode_frame(img, frame_idx)

        if frame_idx > 0 and frame_idx % self.check_interval == 0:
            top_k = self.check_loop_closure(query_desc, frame_idx)
            if top_k:
                self.loop_closures.append((frame_idx, top_k))
                return top_k[0][0], top_k[0][1]

        return None, 0.0

    def process_sequence_realtime(self, sequence_path, max_frames=None, save_results=True):
        image_dir = Path(sequence_path) / "image_0"
        if not image_dir.exists():
            image_dir = Path(sequence_path) / "image_2"

        if not image_dir.exists():
            print(f"Error: Image directory not found at {image_dir}")
            return []

        images = sorted(image_dir.glob("*.png"))
        if not images:
            images = sorted(image_dir.glob("*.jpg"))

        print(f"Found {len(images)} images in {image_dir}")
        print("Configuration:")
        print("  - Similarity metric: l2 (FAISS IndexFlatL2)")
        print(f"  - Similarity threshold: {self.similarity_threshold}")
        print(f"  - Min time difference: {self.min_time_diff}")
        print(f"  - Check interval: Every {self.check_interval} frames")
        print(f"  - Device: {self.device}")

        if max_frames:
            images = images[:max_frames]

        print("\nStarting real-time processing...")
        for idx, img_path in enumerate(tqdm(images, desc="Processing frames")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            self.process_frame(img, idx)
            self.frame_count = idx + 1

        self.print_statistics()

        if save_results:
            self.save_results(sequence_path)

        return self.loop_closures

    def print_statistics(self):
        print(f"\n{'='*60}")
        print("PROCESSING STATISTICS")
        print(f"{'='*60}")

        print(f"\nFrames processed: {self.frame_count}")
        print(f"Database size: {len(self.faiss_ids)} descriptors")
        print(f"Loop closures detected: {len(self.loop_closures)}")

        if self.encoding_times:
            print("\nEncoding performance:")
            print(f"  - Average time: {np.mean(self.encoding_times):.4f}s")
            print(f"  - Min time: {np.min(self.encoding_times):.4f}s")
            print(f"  - Max time: {np.max(self.encoding_times):.4f}s")

        if self.matching_times:
            print("\nMatching performance:")
            print(f"  - Average time: {np.mean(self.matching_times):.4f}s")
            print(f"  - Min time: {np.min(self.matching_times):.4f}s")
            print(f"  - Max time: {np.max(self.matching_times):.4f}s")
            print(f"  - Total matching operations: {len(self.matching_times)}")

        if self.similarity_scores:
            print("\nSimilarity scores (l2):")
            print(f"  - Min: {min(self.similarity_scores):.3f}")
            print(f"  - Max: {max(self.similarity_scores):.3f}")
            print(f"  - Mean: {np.mean(self.similarity_scores):.3f}")
            print(f"  - Std: {np.std(self.similarity_scores):.3f}")
            print(f"  - Percentiles: 25%={np.percentile(self.similarity_scores, 25):.3f}, "
                  f"50%={np.percentile(self.similarity_scores, 50):.3f}, "
                  f"75%={np.percentile(self.similarity_scores, 75):.3f}")

        if self.loop_closures:
            print("\nTop 5 loop closures by best score:")
            all_best = []
            for q, top_k in self.loop_closures:
                if top_k:
                    m, s = top_k[0]
                    all_best.append((q, m, s))
            all_best.sort(key=lambda x: x[2], reverse=True)
            for i, (q, m, s) in enumerate(all_best[:5], 1):
                print(f"  {i}. Frame {q} -> {m}: score={s:.3f}")

    def save_results(self, sequence_path):
        output_file = Path(sequence_path) / "loop_closures_realtime_l2.txt"
        with open(output_file, 'w') as f:
            f.write("# Real-time Loop Closure Detection Results (Top-K=5)\n")
            f.write(f"# Check interval: every {self.check_interval} frames\n")
            f.write("# Similarity metric: l2\n")
            f.write(f"# Similarity threshold: {self.similarity_threshold}\n")
            f.write(f"# Min time difference: {self.min_time_diff}\n")
            f.write(f"# Total frames processed: {self.frame_count}\n")
            f.write(f"# Total loop closures: {len(self.loop_closures)}\n")
            f.write("# Format: Query_Index Match_Index1 Similarity_Score ... Match_Index5 Similarity_Score\n")
            f.write("#" + "=" * 50 + "\n")

            for query_idx, top_k in self.loop_closures:
                line = f"{query_idx}"
                for i in range(5):
                    if i < len(top_k):
                        m, s = top_k[i]
                        line += f" {m} {s:.6f}"
                    else:
                        line += " -1 0.000000"
                f.write(line + "\n")

        print(f"\nResults saved to: {output_file}")


def main():
    SEQUENCE_PATH = "/home/wzb/0000000000alldatas/dataset/sequences/00"
    SUPERPOINT_PATH = "/home/wzb/0000000000alldatas/superpoint.pth"
    # config
    CHECK_INTERVAL = 5
    SIMILARITY_THRESHOLD = 0.5
    MIN_TIME_DIFF = 100
    MAX_FRAMES = None
    DEBUG = False

    detector = RealtimeLoopClosureDetector(
        superpoint_path=SUPERPOINT_PATH,
        similarity_threshold=SIMILARITY_THRESHOLD,
        min_time_diff=MIN_TIME_DIFF,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        debug=DEBUG,
        check_interval=CHECK_INTERVAL
    )

    print("Starting real-time loop closure detection")
    print(f"Processing sequence: {SEQUENCE_PATH}")
    print(f"{'=' * 60}")

    t0 = time.time()

    loop_closures = detector.process_sequence_realtime(
        SEQUENCE_PATH,
        max_frames=MAX_FRAMES,
        save_results=True
    )

    total_time = time.time() - t0

    print(f"\n{'=' * 60}")
    print("PROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average FPS: {detector.frame_count / total_time:.2f}")
    print(f"Detected {len(loop_closures)} loop closures")


if __name__ == "__main__":
    main()
