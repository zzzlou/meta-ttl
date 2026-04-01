"""
Parse WebArena evaluation logs and/or results JSON files.
"""
import re
import sys
import json
from collections import defaultdict
from pathlib import Path

ID_DOMAINS = {
    'gitlab':   [43, 44, 49, 62, 81, 96, 99, 112, 122, 156],
    'map':      [1, 3, 7, 19, 20, 38, 52, 53, 69, 91],
    'shopping': [28, 29, 42, 46, 55, 78, 89, 103, 104, 118],
}
OOD_DOMAINS = {
    'reddit':         [5, 14, 97, 98, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 151, 152, 153, 154],
    'shopping_admin': [0, 2, 8, 13, 17, 21, 26, 27, 30, 32, 33, 40, 47, 48, 50, 51, 59, 70, 80, 88, 101, 106, 107, 108, 110, 113, 114, 115, 116, 125, 126, 145, 149, 150, 160],
}
ALL_DOMAINS = {**ID_DOMAINS, **OOD_DOMAINS}

TASK_TO_DOMAIN = {}
for d, tids in ALL_DOMAINS.items():
    for t in tids:
        TASK_TO_DOMAIN[t] = d


def short_meta(meta_str):
    if 'gpt-5' in meta_str:
        return 'gpt5'
    if 'glm-5' in meta_str:
        return 'glm5'
    if 'gemini' in meta_str:
        return 'gemini3flash'
    return meta_str


def infer_split_from_task_ids(task_ids):
    task_set = set(task_ids)
    id_set = set(t for tids in ID_DOMAINS.values() for t in tids)
    ood_set = set(t for tids in OOD_DOMAINS.values() for t in tids)
    if task_set and task_set.issubset(id_set):
        return 'ID'
    if task_set and task_set.issubset(ood_set):
        return 'OOD'
    return 'UNKNOWN'


def infer_prompt_from_path(path_str):
    s = path_str.lower()
    if 'opt_2' in s or '_opt_' in s and 'glm5' in s:
        return 'OPT_2'
    if 'opt_3' in s or '_opt_' in s and 'gpt5' in s:
        return 'OPT_3'
    if 'opt_4' in s or '_opt_' in s and 'gemini3flash' in s:
        return 'OPT_4'
    if 'default' in s:
        return 'DEFAULT'
    return 'UNKNOWN'


def calculate_w_auc(scores, max_score=1.0):
    if not scores:
        return 0.0
    total_weight = sum(range(1, len(scores) + 1))
    weighted_sum = sum((i + 1) * score for i, score in enumerate(scores))
    return float(weighted_sum / (total_weight * max_score))


def parse_json_results(filepath):
    with open(filepath) as f:
        data = json.load(f)

    if not isinstance(data, dict) or 'results' not in data:
        raise ValueError(f"Unsupported JSON format in {filepath}")

    meta = data.get('metadata', {})
    results = data.get('results', [])
    task_ids = [entry['task_id'] for entry in results if 'task_id' in entry]
    split = infer_split_from_task_ids(task_ids)

    prompt_sources = [
        str(filepath),
        str(meta.get('log_dir', '')),
        str(data.get('merge_metadata', {}).get('original_results', '')),
        str(data.get('merge_metadata', {}).get('rerun_results', '')),
    ]
    prompt = next((p for p in (infer_prompt_from_path(s) for s in prompt_sources) if p != 'UNKNOWN'), 'UNKNOWN')

    task_results = {}
    task_avgs = {}
    task_maxs = {}
    error_tids = set()

    for entry in results:
        if 'task_id' not in entry:
            continue
        tid = int(entry['task_id'])
        episodes = entry.get('episodes', [])
        traj_vals = [float(ep.get('score', 0.0)) for ep in episodes]
        task_results[tid] = calculate_w_auc(traj_vals)
        task_avgs[tid] = sum(traj_vals) / len(traj_vals) if traj_vals else 0.0
        task_maxs[tid] = max(traj_vals) if traj_vals else 0.0

        if any('error' in ep and ep.get('error') for ep in episodes):
            error_tids.add(tid)

    return [{
        'meta': short_meta(str(meta.get('meta_model', '?'))),
        'actor': short_meta(str(meta.get('actor_model', '?'))),
        'prompt': prompt,
        'split': split,
        'results': task_results,
        'avgs': task_avgs,
        'maxs': task_maxs,
        'errors': error_tids,
        'global_wauc': None,
    }]


def parse_log(filepath):
    with open(filepath) as f:
        text = f.read()

    headers = list(re.finditer(
        r'  META: (.+)\n  ACTOR: (.+)\n  PROMPT: (\S+)\n  SPLIT: (\S+)', text
    ))

    tables = list(re.finditer(
        r'WebArena Meta-Agent Results \| Meta: [^\n]+\n'
        r'Tasks: \d+ \| Max Episodes: \d+ \| Max Score: \d+\n'
        r'={80}\n'
        r'Task ID.*?Max Score\s*\n'
        r'-{70,80}\n'
        r'(.*?)\n-{70,80}\n'
        r'ALL\s+GLOBAL\s+=+\s+([\d.]+)',
        text, re.DOTALL
    ))

    run_blocks = list(re.finditer(
        r'Saving raw results.*?\n(.*?)Results saved to:', text, re.DOTALL
    ))

    groups = []
    for i, h in enumerate(headers):
        meta, actor, prompt, split = h.group(1).strip(), h.group(2).strip(), h.group(3), h.group(4)
        ms = short_meta(meta)

        task_results = {}
        task_avgs = {}
        task_maxs = {}
        if i < len(tables):
            for m in re.finditer(r'Task (\d+)\s+r0\s+\[([^\]]+)\]\s+([\d.]+)\s+([\d.]+)', tables[i].group(1)):
                tid = int(m.group(1))
                task_results[tid] = float(m.group(3))
                traj_vals = [float(x.strip()) for x in m.group(2).split(',')]
                task_avgs[tid] = sum(traj_vals) / len(traj_vals) if traj_vals else 0.0
                task_maxs[tid] = float(m.group(4))

        error_tids = set()
        if i < len(run_blocks):
            for m in re.finditer(r'Task (\d+) \(r0\): FAIL traj=\[0\].*?ERROR', run_blocks[i].group(1)):
                error_tids.add(int(m.group(1)))

        groups.append({
            'meta': ms,
            'actor': short_meta(actor),
            'prompt': prompt,
            'split': split,
            'results': task_results,
            'avgs': task_avgs,
            'maxs': task_maxs,
            'errors': error_tids,
            'global_wauc': float(tables[i].group(2)) if i < len(tables) else None,
        })

    return groups


def parse_input(filepath):
    path = Path(filepath)
    if path.suffix.lower() == '.json':
        return parse_json_results(filepath)
    return parse_log(filepath)


def _fmt3(w, u, s):
    wf = f"{w:.3f}" if w is not None else "  -  "
    uf = f"{u:.3f}" if u is not None else "  -  "
    sf = f"{s:.3f}" if s is not None else "  -  "
    return f"{wf}/{uf}/{sf}"


def domain_avg(results, domain_tids, exclude=None):
    exclude = exclude or set()
    scores = [results[t] for t in domain_tids if t in results and t not in exclude]
    return (sum(scores) / len(scores), len(scores)) if scores else (None, 0)


def print_results(groups):
    pair_map = defaultdict(dict)
    for g in groups:
        pair_map[(g['meta'], g['prompt'])][g['split']] = g

    print("\n" + "=" * 130)
    print("ID Results — per-domain metrics: W-AUC / AvgScore / SR")
    print("=" * 130)
    print(f"{'Group':<24} {'---gitlab---':>17} {'----map-----':>17} {'--shopping--':>17} {'----ID avg--':>17}")
    print(f"{'':24} {'w/u/sr':>17} {'w/u/sr':>17} {'w/u/sr':>17} {'w/u/sr':>17}")
    print("-" * 130)

    for (ms, prompt), splits in sorted(pair_map.items()):
        if 'ID' not in splits:
            continue
        g = splits['ID']
        res, task_avgs_map, task_maxs_map = g['results'], g['avgs'], g['maxs']
        w_avgs, u_avgs, sr_avgs = {}, {}, {}
        for d in ['gitlab', 'map', 'shopping']:
            w_avgs[d], _ = domain_avg(res, ID_DOMAINS[d])
            u_avgs[d], _ = domain_avg(task_avgs_map, ID_DOMAINS[d])
            sr_avgs[d], _ = domain_avg(task_maxs_map, ID_DOMAINS[d])
        id_w = sum(v for v in w_avgs.values() if v is not None) / sum(1 for v in w_avgs.values() if v is not None)
        id_u = sum(v for v in u_avgs.values() if v is not None) / sum(1 for v in u_avgs.values() if v is not None)
        id_sr = sum(v for v in sr_avgs.values() if v is not None) / sum(1 for v in sr_avgs.values() if v is not None)

        cols = [_fmt3(w_avgs[d], u_avgs[d], sr_avgs[d]) for d in ['gitlab', 'map', 'shopping']]
        cols.append(_fmt3(id_w, id_u, id_sr))
        print(f"{ms + ' ' + prompt:<24} {cols[0]:>17} {cols[1]:>17} {cols[2]:>17} {cols[3]:>17}")

    print("\n" + "=" * 130)
    print("ID Results — cleaned (exclude tasks that errored in EITHER run of a pair)")
    print("=" * 130)
    print(f"{'Group':<24} {'---gitlab---':>17} {'----map-----':>17} {'--shopping--':>17} {'----ID avg--':>17} {'n_gitlab':>9} {'n_map':>7} {'n_shop':>8}")
    print(f"{'':24} {'w/u/sr':>17} {'w/u/sr':>17} {'w/u/sr':>17} {'w/u/sr':>17}")
    print("-" * 130)

    metas = sorted({g['meta'] for g in groups})
    for ms in metas:
        def_g = pair_map.get((ms, 'DEFAULT'), {}).get('ID')
        if not def_g:
            continue

        opt_prompts = sorted(
            prompt for meta, prompt in pair_map.keys()
            if meta == ms and prompt != 'DEFAULT' and 'ID' in pair_map[(meta, prompt)]
        )
        for opt_p in opt_prompts:
            opt_g = pair_map.get((ms, opt_p), {}).get('ID')
            if not opt_g:
                continue

            union_errors = def_g['errors'] | opt_g['errors']

            for g, label in [(def_g, f"{ms} DEFAULT"), (opt_g, f"{ms} {opt_p}")]:
                w_avgs, u_avgs, sr_avgs, ns = {}, {}, {}, {}
                for d in ['gitlab', 'map', 'shopping']:
                    w_avgs[d], ns[d] = domain_avg(g['results'], ID_DOMAINS[d], exclude=union_errors)
                    u_avgs[d], _ = domain_avg(g['avgs'], ID_DOMAINS[d], exclude=union_errors)
                    sr_avgs[d], _ = domain_avg(g['maxs'], ID_DOMAINS[d], exclude=union_errors)
                valid_w = [v for v in w_avgs.values() if v is not None]
                valid_u = [v for v in u_avgs.values() if v is not None]
                valid_sr = [v for v in sr_avgs.values() if v is not None]
                id_w = sum(valid_w) / len(valid_w) if valid_w else 0
                id_u = sum(valid_u) / len(valid_u) if valid_u else 0
                id_sr = sum(valid_sr) / len(valid_sr) if valid_sr else 0

                cols = [_fmt3(w_avgs[d], u_avgs[d], sr_avgs[d]) for d in ['gitlab', 'map', 'shopping']]
                cols.append(_fmt3(id_w, id_u, id_sr))
                ns_strs = [f"{ns[d]:>2d}/{len(ID_DOMAINS[d])}" for d in ['gitlab', 'map', 'shopping']]
                print(f"{label:<24} {cols[0]:>17} {cols[1]:>17} {cols[2]:>17} {cols[3]:>17} {ns_strs[0]:>9} {ns_strs[1]:>7} {ns_strs[2]:>8}")
            print("-" * 130)

    print("\n" + "=" * 130)
    print("OOD Results — raw, per-domain metrics: W-AUC / AvgScore / SR")
    print("=" * 130)
    print(f"{'Group':<24} {'---reddit---':>17} {'--shop_admin':>17} {'---OOD avg--':>17} {'errors':>7}")
    print(f"{'':24} {'w/u/sr':>17} {'w/u/sr':>17} {'w/u/sr':>17}")
    print("-" * 130)

    for (ms, prompt), splits in sorted(pair_map.items()):
        if 'OOD' not in splits:
            continue
        g = splits['OOD']
        res, task_avgs_map, task_maxs_map, errs = g['results'], g['avgs'], g['maxs'], g['errors']
        w_avgs, u_avgs, sr_avgs = {}, {}, {}
        for d in ['reddit', 'shopping_admin']:
            w_avgs[d], _ = domain_avg(res, OOD_DOMAINS[d])
            u_avgs[d], _ = domain_avg(task_avgs_map, OOD_DOMAINS[d])
            sr_avgs[d], _ = domain_avg(task_maxs_map, OOD_DOMAINS[d])
        ood_w = sum(v for v in w_avgs.values() if v is not None) / sum(1 for v in w_avgs.values() if v is not None)
        ood_u = sum(v for v in u_avgs.values() if v is not None) / sum(1 for v in u_avgs.values() if v is not None)
        ood_sr = sum(v for v in sr_avgs.values() if v is not None) / sum(1 for v in sr_avgs.values() if v is not None)

        cols = [_fmt3(w_avgs[d], u_avgs[d], sr_avgs[d]) for d in ['reddit', 'shopping_admin']]
        cols.append(_fmt3(ood_w, ood_u, ood_sr))
        print(f"{ms + ' ' + prompt:<24} {cols[0]:>17} {cols[1]:>17} {cols[2]:>17} {len(errs):>4d}/54")

    print("\n" + "=" * 130)
    print("OOD Results — cleaned (exclude tasks that errored in EITHER run of a pair)")
    print("=" * 130)
    print(f"{'Group':<24} {'---reddit---':>17} {'--shop_admin':>17} {'---OOD avg--':>17} {'n_reddit':>9} {'n_sadm':>7}")
    print(f"{'':24} {'w/u/sr':>17} {'w/u/sr':>17} {'w/u/sr':>17}")
    print("-" * 130)

    for ms in metas:
        def_g = pair_map.get((ms, 'DEFAULT'), {}).get('OOD')
        if not def_g:
            continue

        opt_prompts = sorted(
            prompt for meta, prompt in pair_map.keys()
            if meta == ms and prompt != 'DEFAULT' and 'OOD' in pair_map[(meta, prompt)]
        )
        for opt_p in opt_prompts:
            opt_g = pair_map.get((ms, opt_p), {}).get('OOD')
            if not opt_g:
                continue

            union_errors = def_g['errors'] | opt_g['errors']

            for g, label in [(def_g, f"{ms} DEFAULT"), (opt_g, f"{ms} {opt_p}")]:
                w_avgs, u_avgs, sr_avgs, ns = {}, {}, {}, {}
                for d in ['reddit', 'shopping_admin']:
                    w_avgs[d], ns[d] = domain_avg(g['results'], OOD_DOMAINS[d], exclude=union_errors)
                    u_avgs[d], _ = domain_avg(g['avgs'], OOD_DOMAINS[d], exclude=union_errors)
                    sr_avgs[d], _ = domain_avg(g['maxs'], OOD_DOMAINS[d], exclude=union_errors)
                valid_w = [v for v in w_avgs.values() if v is not None]
                valid_u = [v for v in u_avgs.values() if v is not None]
                valid_sr = [v for v in sr_avgs.values() if v is not None]
                ood_w = sum(valid_w) / len(valid_w) if valid_w else 0
                ood_u = sum(valid_u) / len(valid_u) if valid_u else 0
                ood_sr = sum(valid_sr) / len(valid_sr) if valid_sr else 0

                cols = [_fmt3(w_avgs[d], u_avgs[d], sr_avgs[d]) for d in ['reddit', 'shopping_admin']]
                cols.append(_fmt3(ood_w, ood_u, ood_sr))
                ns_strs = [f"{ns[d]:>2d}/{len(OOD_DOMAINS[d])}" for d in ['reddit', 'shopping_admin']]
                print(f"{label:<24} {cols[0]:>17} {cols[1]:>17} {cols[2]:>17} {ns_strs[0]:>9} {ns_strs[1]:>7}")
            print("-" * 130)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <input_file> [<input_file> ...]")
        sys.exit(1)

    groups = []
    for path in sys.argv[1:]:
        parsed = parse_input(path)
        groups.extend(parsed)
        print(f"Parsed {len(parsed)} evaluation groups from {path}")
    print_results(groups)
