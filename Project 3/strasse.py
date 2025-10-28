import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

# - Use original 4x4 ss-blocks a11..a44 and b11..b44 (views, no precomputed combos)
# - Top level: 7 Strassen products (groups). Each group is expanded by another Strassen
#   into 7 base multiplications => total 49 base multiplications (P{group}{1..7}).
# - Communication accounting: each task (core) preallocates one a-block and one b-block.
#   First use of any other original block in a task adds one block-size to communication.
# - Visualization: generate PNG figures for communication per group/task and block usage frequency.


def build_group_maps(A_quads, B_quads):
    # Return per-group X/Y composition map
    # Each entry: key in {'11','12','21','22'} maps to a list of (block_name, coeff)
    groups = {}

    # Group 1: (A11 + A22) * (B11 + B22)
    groups[1] = {
        'X': {
            '11': [('a11', +1), ('a33', +1)],
            '12': [('a12', +1), ('a34', +1)],
            '21': [('a21', +1), ('a43', +1)],
            '22': [('a22', +1), ('a44', +1)],
        },
        'Y': {
            '11': [('b11', +1), ('b33', +1)],
            '12': [('b12', +1), ('b34', +1)],
            '21': [('b21', +1), ('b43', +1)],
            '22': [('b22', +1), ('b44', +1)],
        }
    }

    # Group 2: (A21 + A22) * B11
    groups[2] = {
        'X': {
            '11': [('a31', +1), ('a33', +1)],
            '12': [('a32', +1), ('a34', +1)],
            '21': [('a41', +1), ('a43', +1)],
            '22': [('a42', +1), ('a44', +1)],
        },
        'Y': {
            '11': [('b11', +1)],
            '12': [('b12', +1)],
            '21': [('b21', +1)],
            '22': [('b22', +1)],
        }
    }

    # Group 3: A11 * (B12 - B22)
    groups[3] = {
        'X': {
            '11': [('a11', +1)],
            '12': [('a12', +1)],
            '21': [('a21', +1)],
            '22': [('a22', +1)],
        },
        'Y': {
            '11': [('b13', +1), ('b33', -1)],
            '12': [('b14', +1), ('b34', -1)],
            '21': [('b23', +1), ('b43', -1)],
            '22': [('b24', +1), ('b44', -1)],
        }
    }

    # Group 4: A22 * (B21 - B11)
    groups[4] = {
        'X': {
            '11': [('a33', +1)],
            '12': [('a34', +1)],
            '21': [('a43', +1)],
            '22': [('a44', +1)],
        },
        'Y': {
            '11': [('b31', +1), ('b11', -1)],
            '12': [('b32', +1), ('b12', -1)],
            '21': [('b41', +1), ('b21', -1)],
            '22': [('b42', +1), ('b22', -1)],
        }
    }

    # Group 5: (A11 + A12) * B22
    groups[5] = {
        'X': {
            '11': [('a11', +1), ('a13', +1)],
            '12': [('a12', +1), ('a14', +1)],
            '21': [('a21', +1), ('a23', +1)],
            '22': [('a22', +1), ('a24', +1)],
        },
        'Y': {
            '11': [('b33', +1)],
            '12': [('b34', +1)],
            '21': [('b43', +1)],
            '22': [('b44', +1)],
        }
    }

    # Group 6: (A21 - A11) * (B11 + B12)
    groups[6] = {
        'X': {
            '11': [('a31', +1), ('a11', -1)],
            '12': [('a32', +1), ('a12', -1)],
            '21': [('a41', +1), ('a21', -1)],
            '22': [('a42', +1), ('a22', -1)],
        },
        'Y': {
            '11': [('b11', +1), ('b13', +1)],
            '12': [('b12', +1), ('b14', +1)],
            '21': [('b21', +1), ('b23', +1)],
            '22': [('b22', +1), ('b24', +1)],
        }
    }

    # Group 7: (A12 - A22) * (B21 + B22)
    groups[7] = {
        'X': {
            '11': [('a13', +1), ('a33', -1)],
            '12': [('a14', +1), ('a34', -1)],
            '21': [('a23', +1), ('a43', -1)],
            '22': [('a24', +1), ('a44', -1)],
        },
        'Y': {
            '11': [('b31', +1), ('b33', +1)],
            '12': [('b32', +1), ('b34', +1)],
            '21': [('b41', +1), ('b43', +1)],
            '22': [('b42', +1), ('b44', +1)],
        }
    }

    return groups


def strassen_second_level_tasks(group_id):
    # Return the 7 base Strassen tasks for the group
    # Each task: (name, needX_keys, sx, needY_keys, sy) with keys in {'11','12','21','22'}
    label = f'P{group_id}'
    tasks = []
    tasks.append((f'{label}1', ['11', '22'], [1, 1], ['11', '22'], [1, 1]))
    tasks.append((f'{label}2', ['21', '22'], [1, 1], ['11'], [1]))
    tasks.append((f'{label}3', ['11'], [1], ['12', '22'], [1, -1]))
    tasks.append((f'{label}4', ['22'], [1], ['21', '11'], [1, -1]))
    tasks.append((f'{label}5', ['11', '12'], [1, 1], ['22'], [1]))
    tasks.append((f'{label}6', ['21', '11'], [1, -1], ['11', '12'], [1, 1]))
    tasks.append((f'{label}7', ['12', '22'], [1, -1], ['21', '22'], [1, 1]))
    return tasks


def exec_task_strict(name, needX, sx, needY, sy, groups_maps, a_blocks, b_blocks, pre_a, pre_b, block_bytes):
    # Build S and T from original blocks, and account communication cost
    gid = int(name[1])  # name like 'P11' -> group = 1
    Xmap = groups_maps[gid]['X']
    Ymap = groups_maps[gid]['Y']

    S = None
    T = None
    used_a = set()
    used_b = set()
    comm = 0

    # Assemble S
    for i, k in enumerate(needX):
        combo = Xmap[k]
        part = np.zeros_like(next(iter(a_blocks.values())))
        for blk, coef in combo:
            if blk != pre_a and blk not in used_a:
                comm += block_bytes
                used_a.add(blk)
            part = part + (coef * a_blocks[blk])
        if S is None:
            S = part if sx[i] == 1 else -part
        else:
            S = S + part if sx[i] == 1 else S - part

    # Assemble T
    for j, k in enumerate(needY):
        combo = Ymap[k]
        part = np.zeros_like(next(iter(b_blocks.values())))
        for blk, coef in combo:
            if blk != pre_b and blk not in used_b:
                comm += block_bytes
                used_b.add(blk)
            part = part + (coef * b_blocks[blk])
        if T is None:
            T = part if sy[j] == 1 else -part
        else:
            T = T + part if sy[j] == 1 else T - part

    P = S @ T
    return name, P, comm


if __name__ == '__main__':
    t_all_start = time.time()

    # Config
    N = 1024
    np.random.seed(0)
    A = np.random.randn(N, N).astype(np.float64)
    B = np.random.randn(N, N).astype(np.float64)

    n2 = N // 2
    n4 = N // 4
    rows = [(0, n4), (n4, n2), (n2, n2 + n4), (n2 + n4, N)]
    cols = [(0, n4), (n4, n2), (n2, n2 + n4), (n2 + n4, N)]

    # Original ss-block views (no precomputed combos)
    a = {}
    b = {}
    for i in range(4):
        for j in range(4):
            a[f'a{i+1}{j+1}'] = A[rows[i][0]:rows[i][1], cols[j][0]:cols[j][1]]
            b[f'b{i+1}{j+1}'] = B[rows[i][0]:rows[i][1], cols[j][0]:cols[j][1]]

    # Top-level 2x2 submatrix dictionaries (for readability)
    A_quads = {
        'A11': {'11': a['a11'], '12': a['a12'], '21': a['a21'], '22': a['a22']},
        'A12': {'11': a['a13'], '12': a['a14'], '21': a['a23'], '22': a['a24']},
        'A21': {'11': a['a31'], '12': a['a32'], '21': a['a41'], '22': a['a42']},
        'A22': {'11': a['a33'], '12': a['a34'], '21': a['a43'], '22': a['a44']},
    }
    B_quads = {
        'B11': {'11': b['b11'], '12': b['b12'], '21': b['b21'], '22': b['b22']},
        'B12': {'11': b['b13'], '12': b['b14'], '21': b['b23'], '22': b['b24']},
        'B21': {'11': b['b31'], '12': b['b32'], '21': b['b41'], '22': b['b42']},
        'B22': {'11': b['b33'], '12': b['b34'], '21': b['b43'], '22': b['b44']},
    }

    # Group composition maps (index-only, no materialized combos)
    groups_maps = build_group_maps(A_quads, B_quads)

    # Create 49 tasks (P{group}{1..7})
    all_tasks = []
    for gid in range(1, 8):
        all_tasks.extend(strassen_second_level_tasks(gid))

    # Frequency statistics per group for original blocks (used for prealloc selection)
    group_freq_a = {gid: {} for gid in range(1, 8)}
    group_freq_b = {gid: {} for gid in range(1, 8)}
    for gid in range(1, 8):
        Xmap = groups_maps[gid]['X']
        Ymap = groups_maps[gid]['Y']
        for k in ['11', '12', '21', '22']:
            for blk, _ in Xmap[k]:
                group_freq_a[gid][blk] = group_freq_a[gid].get(blk, 0) + 1
            for blk, _ in Ymap[k]:
                group_freq_b[gid][blk] = group_freq_b[gid].get(blk, 0) + 1

    # Block size in bytes
    block_bytes = a['a11'].nbytes

    # Execute 49 base multiplications in parallel (strict communication accounting)
    t_compute_start = time.time()
    results = {}
    total_comm = 0

    def choose_prealloc(name, needX, needY):
        # Choose one a-block and one b-block among task-used blocks with highest group frequency
        gid = int(name[1])
        Xmap = groups_maps[gid]['X']
        Ymap = groups_maps[gid]['Y']
        a_used = []
        b_used = []
        for k in needX:
            a_used.extend([blk for blk, _ in Xmap[k]])
        for k in needY:
            b_used.extend([blk for blk, _ in Ymap[k]])
        pre_a = max(a_used, key=lambda z: group_freq_a[gid].get(z, 0)) if a_used else list(group_freq_a[gid].keys())[0]
        pre_b = max(b_used, key=lambda z: group_freq_b[gid].get(z, 0)) if b_used else list(group_freq_b[gid].keys())[0]
        return pre_a, pre_b

    with ThreadPoolExecutor(max_workers=49) as ex:
        futs = []
        for name, needX, sx, needY, sy in all_tasks:
            pre_a, pre_b = choose_prealloc(name, needX, needY)
            futs.append(
                ex.submit(
                    exec_task_strict,
                    name, needX, sx, needY, sy,
                    groups_maps, a, b, pre_a, pre_b, block_bytes
                )
            )
        for f in as_completed(futs):
            name, P, comm = f.result()
            results[name] = (P, comm)
            total_comm += comm
    t_compute_end = time.time()

    # Second-level regroup: 7 P's -> one 2x2 submatrix per group
    def gather_group(gid):
        D = {i: results[f'P{gid}{i}'][0] for i in range(1, 8)}
        Z11 = D[1] + D[4] - D[5] + D[7]
        Z12 = D[3] + D[5]
        Z21 = D[2] + D[4]
        Z22 = D[1] - D[2] + D[3] + D[6]
        return {'11': Z11, '12': Z12, '21': Z21, '22': Z22}

    G = [gather_group(i) for i in range(1, 8)]

    # Top-level regroup: assemble C quadrants
    C11 = {k: G[0][k] + G[3][k] - G[4][k] + G[6][k] for k in ['11', '12', '21', '22']}
    C12 = {k: G[2][k] + G[4][k] for k in ['11', '12', '21', '22']}
    C21 = {k: G[1][k] + G[3][k] for k in ['11', '12', '21', '22']}
    C22 = {k: G[0][k] - G[1][k] + G[2][k] + G[5][k] for k in ['11', '12', '21', '22']}

    # Stitch final C matrix from 4x4 ss-block layout
    C = np.empty_like(A)
    # C11: rows 0,1; cols 0,1
    C[rows[0][0]:rows[0][1], cols[0][0]:cols[0][1]] = C11['11']
    C[rows[0][0]:rows[0][1], cols[1][0]:cols[1][1]] = C11['12']
    C[rows[1][0]:rows[1][1], cols[0][0]:cols[0][1]] = C11['21']
    C[rows[1][0]:rows[1][1], cols[1][0]:cols[1][1]] = C11['22']
    # C12: rows 0,1; cols 2,3
    C[rows[0][0]:rows[0][1], cols[2][0]:cols[2][1]] = C12['11']
    C[rows[0][0]:rows[0][1], cols[3][0]:cols[3][1]] = C12['12']
    C[rows[1][0]:rows[1][1], cols[2][0]:cols[2][1]] = C12['21']
    C[rows[1][0]:rows[1][1], cols[3][0]:cols[3][1]] = C12['22']
    # C21: rows 2,3; cols 0,1
    C[rows[2][0]:rows[2][1], cols[0][0]:cols[0][1]] = C21['11']
    C[rows[2][0]:rows[2][1], cols[1][0]:cols[1][1]] = C21['12']
    C[rows[3][0]:rows[3][1], cols[0][0]:cols[0][1]] = C21['21']
    C[rows[3][0]:rows[3][1], cols[1][0]:cols[1][1]] = C21['22']
    # C22: rows 2,3; cols 2,3
    C[rows[2][0]:rows[2][1], cols[2][0]:cols[2][1]] = C22['11']
    C[rows[2][0]:rows[2][1], cols[3][0]:cols[3][1]] = C22['12']
    C[rows[3][0]:rows[3][1], cols[2][0]:cols[2][1]] = C22['21']
    C[rows[3][0]:rows[3][1], cols[3][0]:cols[3][1]] = C22['22']

    # Correctness check
    C_ref = A @ B
    ok = np.allclose(C, C_ref, atol=1e-8)

    t_all_end = time.time()

    # Visualization (saved PNGs)
    try:
        import matplotlib.pyplot as plt
        out_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()

        # Per-group communication (MB)
        group_comm = []
        for gid in range(1, 8):
            gsum = 0
            for i in range(1, 8):
                gsum += results[f'P{gid}{i}'][1]
            group_comm.append(gsum / (1024*1024))
        plt.figure(figsize=(8, 4))
        plt.bar([f'G{gid}' for gid in range(1, 8)], group_comm, color='#4C78A8')
        plt.title('Communication per Group (MB)')
        plt.xlabel('Group')
        plt.ylabel('MB')
        plt.tight_layout()
        path1 = os.path.join(out_dir, 'comm_per_group.png')
        plt.savefig(path1)
        plt.close()

        # Per-task communication (MB)
        task_names = sorted(results.keys(), key=lambda x: (int(x[1]), int(x[2])))
        task_comm_mb = [results[n][1] / (1024*1024) for n in task_names]
        plt.figure(figsize=(12, 5))
        plt.bar(range(len(task_names)), task_comm_mb, color='#F58518')
        plt.title('Communication per Task (MB)')
        plt.xlabel('Task index (Pij)')
        plt.ylabel('MB')
        plt.xticks(range(len(task_names)), task_names, rotation=90, fontsize=6)
        plt.tight_layout()
        path2 = os.path.join(out_dir, 'comm_per_task.png')
        plt.savefig(path2)
        plt.close()

        # Block usage frequency heatmaps (a- and b-blocks) per group
        a_blocks_list = [f'a{i}{j}' for i in range(1,5) for j in range(1,5)]
        b_blocks_list = [f'b{i}{j}' for i in range(1,5) for j in range(1,5)]
        freq_a = np.zeros((7, 16), dtype=int)
        freq_b = np.zeros((7, 16), dtype=int)
        for gid in range(1, 8):
            for idx, blk in enumerate(a_blocks_list):
                freq_a[gid-1, idx] = group_freq_a[gid].get(blk, 0)
            for idx, blk in enumerate(b_blocks_list):
                freq_b[gid-1, idx] = group_freq_b[gid].get(blk, 0)
        # a-block heatmap
        plt.figure(figsize=(10, 4))
        plt.imshow(freq_a, aspect='auto', cmap='Blues')
        plt.colorbar(label='Frequency')
        plt.title('a-block Usage Frequency per Group')
        plt.xlabel('a-blocks (a11..a44)')
        plt.ylabel('Group')
        plt.yticks(range(7), [f'G{gid}' for gid in range(1,8)])
        plt.xticks(range(16), a_blocks_list, rotation=90, fontsize=6)
        plt.tight_layout()
        path3 = os.path.join(out_dir, 'freq_a_heatmap.png')
        plt.savefig(path3)
        plt.close()
        # b-block heatmap
        plt.figure(figsize=(10, 4))
        plt.imshow(freq_b, aspect='auto', cmap='Oranges')
        plt.colorbar(label='Frequency')
        plt.title('b-block Usage Frequency per Group')
        plt.xlabel('b-blocks (b11..b44)')
        plt.ylabel('Group')
        plt.yticks(range(7), [f'G{gid}' for gid in range(1,8)])
        plt.xticks(range(16), b_blocks_list, rotation=90, fontsize=6)
        plt.tight_layout()
        path4 = os.path.join(out_dir, 'freq_b_heatmap.png')
        plt.savefig(path4)
        plt.close()

        print('Saved figures:')
        print(path1)
        print(path2)
        print(path3)
        print(path4)
    except Exception as e:
        print('Visualization skipped:', repr(e))

    print(f'allclose = {ok}')
    print(f'Total communication (per-core fetch) = {total_comm / (1024*1024):.2f} MB')
    print(f'Average per core = {total_comm / 49 / (1024*1024):.2f} MB')