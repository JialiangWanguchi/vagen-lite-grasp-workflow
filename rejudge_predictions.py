"""Replay the deterministic judge on saved predictions without model generation."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
from task_contract import judge_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--max-tokens', type=int, required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding='utf-8').splitlines()
            if line.strip()]
    predictions = [json.loads(line) for line in Path(args.predictions).read_text(encoding='utf-8').splitlines()
                   if line.strip()]
    by_sha = {hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest(): row
              for row in rows}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rescored = []
    for old in predictions:
        index = int(old['index'])
        row = by_sha.get(old.get('sample_sha256'))
        if row is None:
            if len(predictions) != len(rows):
                raise ValueError('Prediction lacks a resolvable sample hash for a subset evaluation')
            row = rows[index]
        if old.get('task_id') != row.get('task_id'):
            raise ValueError(f'Task mismatch at index {index}')
        judgement = judge_answer(old['prediction'], row,
                                  finish_reason=old.get('finish_reason'),
                                  output_tokens=old.get('output_tokens'),
                                  max_tokens=args.max_tokens)
        rescored.append({**old, **judgement})
    with (output / 'rejudged_predictions.jsonl').open('w', encoding='utf-8') as stream:
        for record in rescored:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    modes = collections.Counter(record['match_mode'] for record in rescored)
    summary = {
        'n': len(rescored),
        'strict_correct': sum(record['exact_match'] for record in rescored),
        'accepted_correct': sum(record['accepted_match'] for record in rescored),
        'format_valid': sum(record['format_valid'] for record in rescored),
        'hard_negative_length': sum(record['hard_negative_length'] for record in rescored),
        'review_required': sum(record['review_required'] for record in rescored),
        'match_modes': dict(sorted(modes.items())),
    }
    (output / 'rejudged_metrics.json').write_text(json.dumps(summary, indent=2) + '\n',
                                                   encoding='utf-8')
    candidates = [record for record in rescored
                  if record['fallback_applied'] or record['review_required']]
    with (output / 'manual_audit_candidates.jsonl').open('w', encoding='utf-8') as stream:
        for record in candidates:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
