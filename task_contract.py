"""Shared, backend-independent prompt and evaluation contract for all arms."""
import json
import re


def ground_truth(row):
    value = row['gt_answer']
    if row['task_id'] == 'A2':
        if not isinstance(value, str) or value not in list('ABCDE'):
            raise ValueError('Invalid A2 ground truth')
    elif row['task_id'] == 'P1':
        value = json.loads(value) if isinstance(value, str) else value
        if not isinstance(value, list) or sorted(value) != list('ABC'):
            raise ValueError('Invalid P1 ground truth')
    else:
        raise ValueError('Unsupported task')
    return value


def evaluate_answer(text, row):
    """Exact JSON object, no fences; P1 answer must be an array, not a string."""
    try:
        def no_duplicates(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('Duplicate JSON key')
                result[key] = value
            return result
        value = json.loads(text, object_pairs_hook=no_duplicates)
        if not isinstance(value, dict) or set(value) != {'think', 'answer'}:
            raise ValueError('Expected exactly think and answer')
        if not isinstance(value['think'], str):
            raise ValueError('think must be a string')
        answer = value['answer']
        if row['task_id'] == 'A2':
            valid = isinstance(answer, str) and answer in list('ABCDE')
        else:
            valid = (isinstance(answer, list) and len(answer) == 3
                     and all(isinstance(x, str) for x in answer)
                     and sorted(answer) == list('ABC'))
        if not valid:
            raise ValueError('Illegal answer type or value')
    except (ValueError, TypeError, KeyError):
        return {'format_valid': False, 'exact_match': False, 'reward': 0.0}
    exact = answer == ground_truth(row)
    return {'format_valid': True, 'exact_match': exact, 'reward': float(exact)}


def sft_target(row):
    # The export contains no expert rationale. Empty think avoids invented labels.
    return json.dumps({'think': '', 'answer': ground_truth(row)}, separators=(',', ':'))


def prompt_parts(row):
    """Replace each filename with neutral clip/frame labels plus an image slot.

    Return text/image parts in the original sequence. A later processor loads
    actual images using image_path internally; filenames never enter model text.
    """
    frames = [(clip['clip_label'], i + 1, frame['image_path'])
              for clip in row['clips'] for i, frame in enumerate(clip['frames'])]
    question = row['question']
    parts = []
    position = 0
    for label, number, image_path in frames:
        index = question.find(image_path, position)
        if index < 0:
            raise ValueError('Image path missing from question: ' + image_path)
        parts.append({'type': 'text', 'text': question[position:index] + f'Clip {label}, Frame {number}:\n'})
        parts.append({'type': 'image', 'image_path': image_path})
        position = index + len(image_path)
    parts.append({'type': 'text', 'text': question[position:]})
    rendered = ''.join(part['text'] for part in parts if part['type'] == 'text')
    if re.search(r'CASE\d+|\.jpg|data/GraSP', rendered):
        raise ValueError('Source filename leakage remains in rendered prompt')
    return parts


if __name__ == '__main__':
    a2 = {'task_id': 'A2', 'gt_answer': 'A'}
    p1 = {'task_id': 'P1', 'gt_answer': '["B","C","A"]'}
    assert evaluate_answer(sft_target(a2), a2)['reward'] == 1
    assert evaluate_answer(sft_target(p1), p1)['reward'] == 1
    assert evaluate_answer('{"think":"","answer":"B"}', a2)['reward'] == 0
    assert not evaluate_answer('```json\n' + sft_target(a2) + '\n```', a2)['format_valid']
    assert not evaluate_answer('{"think":"","answer":["A","A","C"]}', p1)['format_valid']
    assert not evaluate_answer('{"think":"","answer":"A","answer":"A"}', a2)['format_valid']
    print('TASK_CONTRACT_CHECKS_PASS')
