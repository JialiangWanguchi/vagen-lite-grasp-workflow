"""Audit and materialize case-disjoint splits for GraSP-style JSONL data."""
import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path


SPLITS = ('train', 'val', 'test')


def read_rows(paths):
    rows = []
    for raw in paths:
        path = Path(raw)
        rows.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
                    if line.strip())
    return rows


def case_ids(row):
    found = set()
    for clip in row.get('clips', []):
        for frame in clip.get('frames', []):
            source = f"{frame.get('frame_id', '')} {frame.get('image_path', '')}"
            found.update(re.findall(r'CASE\d+', source, flags=re.IGNORECASE))
    if not found:
        raise ValueError(f"No case identifier found in row {row.get('id', '<unknown>')}")
    return {value.upper() for value in found}


def connected_components(rows):
    parent = {}

    def find(item):
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for row in rows:
        cases = sorted(case_ids(row))
        for case in cases:
            find(case)
        for case in cases[1:]:
            union(cases[0], case)
    groups = collections.defaultdict(list)
    for case in parent:
        groups[find(case)].append(case)
    return sorted((sorted(group) for group in groups.values()), key=lambda x: (-len(x), x))


def audit(rows):
    task_counts = collections.Counter(row['task_id'] for row in rows)
    cases_per_task = collections.defaultdict(set)
    row_case_histogram = collections.Counter()
    for row in rows:
        ids = case_ids(row)
        cases_per_task[row['task_id']].update(ids)
        row_case_histogram[len(ids)] += 1
    components = connected_components(rows)
    return {
        'rows': len(rows),
        'task_counts': dict(sorted(task_counts.items())),
        'unique_cases': len(set().union(*(case_ids(row) for row in rows))),
        'cases_per_task': {key: len(value) for key, value in sorted(cases_per_task.items())},
        'row_case_count_histogram': {str(k): v for k, v in sorted(row_case_histogram.items())},
        'case_components': len(components),
        'component_sizes': [len(component) for component in components],
        'strict_three_way_split_of_existing_rows_possible': len(components) >= 3,
        'conclusion': ('Existing rows may be assigned by whole connected component.' if len(components) >= 3
                       else 'Existing rows cannot form three non-empty case-disjoint splits; '
                            'partition cases before regenerating questions.'),
    }


def plan_case_manifest(rows, seed, min_cases_per_split=4):
    cases = sorted(set().union(*(case_ids(row) for row in rows)))
    required = 3 * min_cases_per_split
    if len(cases) < required:
        raise ValueError(f'Need at least {required} cases; found {len(cases)}')
    sizes = {'train': min_cases_per_split + len(cases) - required,
             'val': min_cases_per_split, 'test': min_cases_per_split}
    p1_counts = collections.Counter()
    for row in rows:
        if row['task_id'] == 'P1':
            for case in case_ids(row):
                p1_counts[case] += 1
    supported = [case for case in cases if p1_counts[case]]
    if len(supported) < 3:
        raise ValueError('Need at least one P1-capable case in each split')
    rng = random.Random(seed)
    rng.shuffle(supported)
    assignment = {}
    # Reserve one P1-capable case for each split, including the sealed test split.
    for split, case in zip(('test', 'val', 'train'), supported[:3]):
        assignment[case] = split
    remaining = [case for case in cases if case not in assignment]
    rng.shuffle(remaining)
    for split in SPLITS:
        need = sizes[split] - sum(value == split for value in assignment.values())
        for _ in range(need):
            assignment[remaining.pop()] = split
    if remaining:
        raise AssertionError('Case allocation bug')
    payload = {
        'schema_version': 1,
        'seed': seed,
        'policy': 'case-first; minimum four cases per split for four-case A2 generation',
        'case_to_split': dict(sorted(assignment.items())),
        'counts': {split: sum(value == split for value in assignment.values()) for split in SPLITS},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    payload['manifest_sha256'] = hashlib.sha256(canonical).hexdigest()
    return payload


def materialize(rows, manifest, output_dir):
    mapping = manifest['case_to_split']
    grouped = {split: [] for split in SPLITS}
    violations = []
    for row in rows:
        row_cases = case_ids(row)
        missing = sorted(row_cases - mapping.keys())
        destinations = {mapping[case] for case in row_cases if case in mapping}
        if missing or len(destinations) != 1:
            violations.append({'id': row.get('id'), 'cases': sorted(row_cases),
                               'missing_cases': missing, 'destinations': sorted(destinations)})
            continue
        grouped[destinations.pop()].append(row)
    if violations:
        example = violations[:3]
        raise ValueError(f'{len(violations)} rows cross case partitions or use unknown cases; examples={example}')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, values in grouped.items():
        with (out / f'{split}.jsonl').open('w', encoding='utf-8') as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    return {split: len(values) for split, values in grouped.items()}


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    for name in ('audit', 'plan'):
        child = subparsers.add_parser(name)
        child.add_argument('--input', action='append', required=True)
        child.add_argument('--output', required=True)
        if name == 'plan':
            child.add_argument('--seed', type=int, default=20260906)
    build = subparsers.add_parser('materialize')
    build.add_argument('--input', action='append', required=True)
    build.add_argument('--manifest', required=True)
    build.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    rows = read_rows(args.input)
    if args.command == 'audit':
        result = audit(rows)
    elif args.command == 'plan':
        result = plan_case_manifest(rows, args.seed)
    else:
        manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
        result = {'written': materialize(rows, manifest, args.output_dir)}
    if args.command in ('audit', 'plan'):
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
