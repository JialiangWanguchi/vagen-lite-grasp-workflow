"""Choose an output-token cap from normal validation completions only."""
import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[rank]


def describe(values):
    return {
        'n': len(values),
        'mean': statistics.fmean(values) if values else None,
        'median': statistics.median(values) if values else None,
        'p90': percentile(values, 0.90),
        'p95': percentile(values, 0.95),
        'max': max(values) if values else None,
    }


def recommend_cap(values, quantum=512):
    """Return the smallest k*quantum strictly greater than observed maximum."""
    if not values:
        raise ValueError('No normally completed validation outputs')
    if quantum <= 0 or quantum & (quantum - 1):
        raise ValueError('quantum must be a positive power of two')
    return (max(values) // quantum + 1) * quantum


def load_records(specs, split='val'):
    records = []
    for spec in specs:
        if '=' not in spec:
            raise ValueError('--input must be ARM=PATH')
        arm, raw_path = spec.split('=', 1)
        path = Path(raw_path)
        if path.is_dir():
            path = path / f'{split}_predictions.jsonl'
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                record = json.loads(line)
                record['_arm'] = arm
                records.append(record)
    return records


def build_report(records, quantum=512, source_split='val'):
    if source_split == 'test':
        raise ValueError('Do not tune output length on the held-out test split')
    normal = [r for r in records if r.get('finish_reason') != 'length'
              and not r.get('hard_negative_length', False)]
    lengths = [int(r['output_tokens']) for r in normal]
    if not lengths:
        raise ValueError('All calibration outputs hit the length cap')
    groups = {}
    for record in normal:
        for key in (f"arm:{record['_arm']}", f"task:{record.get('task_id', 'unknown')}"):
            groups.setdefault(key, []).append(int(record['output_tokens']))
    return {
        'selection_split': source_split,
        'selection_rule': 'smallest k*quantum strictly greater than max normal output',
        'quantum': quantum,
        'total_outputs': len(records),
        'excluded_length_limited': len(records) - len(normal),
        'normal_outputs': describe(lengths),
        'groups': {key: describe(value) for key, value in sorted(groups.items())},
        'recommended_max_output_tokens': recommend_cap(lengths, quantum),
        'test_set_used_for_selection': False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='append', required=True, help='ARM=prediction file or result dir')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--quantum', type=int, default=512)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = build_report(load_records(args.input, args.split), args.quantum, args.split)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
