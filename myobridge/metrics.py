"""Evaluation metrics on the held-out split.

Geometry is root aligned without Procrustes fitting. Surface contact uses a
fixed 5 mm edge-geodesic neighborhood of the rest mesh, so the metric does not
depend on the predicted or target deformation. Force errors are in estimated
newtons and are reported per region and macro averaged.
"""
import heapq

import numpy as np
import torch


def geodesic_neighborhood(vertices, faces, tolerance=.005):
    """Binary vertex neighborhood within ``tolerance`` metres on the rest mesh."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    neighbors = [{} for _ in vertices]
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            distance = float(np.linalg.norm(vertices[a] - vertices[b]))
            neighbors[a][int(b)] = distance
            neighbors[b][int(a)] = distance
    near = np.zeros((len(vertices), len(vertices)), bool)
    for start in range(len(vertices)):
        best = {start: 0.}
        queue = [(0., start)]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost > best[node] or cost > tolerance:
                continue
            near[start, node] = True
            for other, weight in neighbors[node].items():
                new = cost + weight
                if new <= tolerance and new < best.get(other, float('inf')):
                    best[other] = new
                    heapq.heappush(queue, (new, other))
    return near


def f1_from_counts(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.ones_like(np.asarray(tp, dtype=float)),
                     where=denominator > 0)


class Metrics:
    """Accumulate prediction and target tensors over the test split."""

    def __init__(self, partition, near, device):
        self.partition = partition.to(device)
        self.near = torch.as_tensor(near, device=device, dtype=torch.float32)
        self.scalars = torch.zeros(11, dtype=torch.float64, device=device)
        self.region = torch.zeros((7, 6), dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, prediction, target):
        if 'joints' in prediction:
            pj = prediction['joints'] - prediction['joints'][:, :1]
            tj = target['joints'] - target['joints'][:, :1]
            joint_mask = target['joint_valid']
            joint_error = torch.linalg.vector_norm(pj - tj, dim=-1) * 1000
            self.scalars[0] += torch.where(joint_mask, joint_error, 0).double().sum()
            self.scalars[1] += joint_mask.sum()
            if 'mesh' in prediction:
                pm = prediction['mesh'] - prediction['joints'][:, :1]
                tm = target['mesh'] - target['joints'][:, :1]
                mesh_mask = target['mesh_valid'][:, None].expand(-1, 778)
                mesh_error = torch.linalg.vector_norm(pm - tm, dim=-1) * 1000
                self.scalars[2] += torch.where(mesh_mask, mesh_error, 0).double().sum()
                self.scalars[3] += mesh_mask.sum()

        actual = target['regional_contact'].bool()
        valid = target['force_valid']
        contact_valid = target.get('contact_valid', valid)
        for region in range(7):
            region_valid = valid[:, region]
            truth = actual[:, region]
            if 'regional_contact' in prediction:
                predicted = prediction['regional_contact'][:, region] >= .5
                usable = contact_valid[:, region]
                self.region[region, 0] += (usable & predicted & truth).sum()
                self.region[region, 1] += (usable & predicted & ~truth).sum()
                self.region[region, 2] += (usable & ~predicted & truth).sum()
                self.region[region, 5] += usable.sum()
            if 'regional_force' in prediction:
                self.region[region, 3] += torch.where(
                    region_valid,
                    (prediction['regional_force'][:, region] - target['regional_force'][:, region]).abs(),
                    0).double().sum()
                self.region[region, 4] += region_valid.sum()

        surface_valid = target['surface_valid'][:, None] & (self.partition[None] >= 0)
        truth = target['vertex_contact'].bool() & surface_valid
        if 'vertex_contact' in prediction:
            self.scalars[10] += target['surface_valid'].sum()
            predicted = (prediction['vertex_contact'] >= .5) & surface_valid
            predicted_matches = (predicted.float() @ self.near) > 0
            truth_matches = (truth.float() @ self.near) > 0
            self.scalars[4] += (predicted & predicted_matches).sum()
            self.scalars[5] += predicted.sum()
            self.scalars[6] += (truth & truth_matches).sum()
            self.scalars[7] += truth.sum()
        force_valid = truth & surface_valid
        if 'vertex_force' in prediction:
            self.scalars[8] += torch.where(
                force_valid,
                (prediction['vertex_force'] - target['vertex_force']).abs(), 0).double().sum()
            self.scalars[9] += force_valid.sum()

    def result(self):
        scalars = self.scalars.cpu().numpy()
        region = self.region.cpu().numpy()

        def ratio(numerator, denominator):
            return float(numerator / denominator) if denominator > 0 else None

        scores = f1_from_counts(region[:, 0], region[:, 1], region[:, 2])
        available = region[:, 5] > 0
        maes = np.divide(region[:, 3], region[:, 4], out=np.zeros(7), where=region[:, 4] > 0)
        precision = ratio(scalars[4], scalars[5])
        recall = ratio(scalars[6], scalars[7])
        if scalars[5] == 0 and scalars[7] == 0:
            surface_f1 = 1. if scalars[10] > 0 else None
        elif scalars[5] == 0 or scalars[7] == 0:
            surface_f1 = 0.
        else:
            surface_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        return dict(
            mpjpe_root_relative_mm=ratio(scalars[0], scalars[1]),
            pve_root_relative_mm=ratio(scalars[2], scalars[3]),
            region_contact_macro_f1=float(scores[available].mean()) if available.any() else None,
            region_force_macro_mae_estimated_n=(float(maes[region[:, 4] > 0].mean())
                                                if (region[:, 4] > 0).any() else None),
            surface_contact_f1_5mm_edge_geodesic=surface_f1,
            contact_vertex_force_mae_estimated_n=ratio(scalars[8], scalars[9]),
            per_region_f1=[float(scores[i]) if available[i] else None for i in range(7)],
            per_region_force_mae=[float(maes[i]) if region[i, 4] > 0 else None for i in range(7)],
            coverage=dict(
                valid_joints=int(scalars[1]),
                valid_mesh_vertices=int(scalars[3]),
                valid_region_frames=region[:, 4].astype(int).tolist(),
                valid_region_contact_frames=region[:, 5].astype(int).tolist(),
                predicted_contact_vertices=int(scalars[5]),
                target_contact_vertices=int(scalars[7]),
                contact_force_vertices=int(scalars[9])),
            conventions='root aligned, no Procrustes, fixed rest edge geodesic 5 mm, '
                        'estimated uncalibrated newtons')


def summarize_seeds(results):
    """Mean and sample standard deviation over independent seeds."""
    import statistics

    keys = ('mpjpe_root_relative_mm', 'pve_root_relative_mm', 'region_contact_macro_f1',
            'region_force_macro_mae_estimated_n', 'surface_contact_f1_5mm_edge_geodesic',
            'contact_vertex_force_mae_estimated_n')
    summary = {}
    for key in keys:
        values = [result[key] for result in results if result.get(key) is not None]
        if not values:
            summary[key] = None
        elif len(values) == 1:
            summary[key] = dict(mean=float(values[0]), std=None, n=1)
        else:
            summary[key] = dict(mean=float(statistics.mean(values)),
                                std=float(statistics.stdev(values)), n=len(values))
    return summary
