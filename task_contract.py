"""Shared prompt, answer and length-aware judging contract for every arm."""
import json
import re


STRICT_REWARD = 1.0
FALLBACK_REWARD = 0.5


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


def _loads_no_duplicates(text):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=no_duplicates)


def evaluate_answer(text, row):
    """Historical metric: exact JSON object, exact case, and no fences."""
    try:
        value = _loads_no_duplicates(text)
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
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return {'format_valid': False, 'exact_match': False, 'reward': 0.0}
    exact = answer == ground_truth(row)
    return {'format_valid': True, 'exact_match': exact, 'reward': float(exact)}


def generation_hit_limit(*, finish_reason=None, truncated=False,
                         output_tokens=None, max_tokens=None):
    """Use trusted generation metadata rather than guessing from text length."""
    if bool(truncated) or finish_reason == 'length':
        return True
    return (finish_reason is None and output_tokens is not None and max_tokens is not None
            and int(output_tokens) >= int(max_tokens))


def _strip_complete_fence(text):
    match = re.fullmatch(r'\s*```(?:json)?\s*\n?(.*?)\n?```\s*', text,
                         flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def _normalise_a2(value):
    if not isinstance(value, str):
        return None, False
    compact = re.sub(r'[.。!！?？:：;；,，]+$', '', value.strip()).strip()
    patterns = (
        r'([a-e])',
        r'(?:option|choice|answer(?:\s+is)?|clip)\s*[-:#]?\s*([a-e])',
        r'(?:the\s+)?(?:correct\s+)?(?:option|choice|answer)\s*[-:#]?\s*(?:is\s*)?([a-e])',
    )
    hits = []
    for pattern in patterns:
        match = re.fullmatch(pattern, compact, flags=re.IGNORECASE)
        if match:
            hits.append(match.group(1).upper())
    lower = re.sub(r'\s+', ' ', compact.lower())
    if lower in {
        'none of the above', 'all four are the same', 'all four clips are the same',
        'all four clips belong to the same surgical phase',
        'none of the above all four clips belong to the same surgical phase',
    }:
        hits.append('E')
    ordinal = {
        'the first clip': 'A', 'first clip': 'A',
        'the second clip': 'B', 'second clip': 'B',
        'the third clip': 'C', 'third clip': 'C',
        'the fourth clip': 'D', 'fourth clip': 'D',
    }
    if lower in ordinal:
        hits.append(ordinal[lower])
    unique = sorted(set(hits))
    return (unique[0], False) if len(unique) == 1 else (None, len(unique) > 1)


def _normalise_p1(value):
    if isinstance(value, list):
        if len(value) == 3 and all(isinstance(x, str) for x in value):
            answer = [x.strip().upper() for x in value]
            return (answer, False) if sorted(answer) == list('ABC') else (None, False)
        return None, False
    if not isinstance(value, str):
        return None, False
    compact = value.strip().rstrip('.。')
    compact = re.sub(r'^(?:answer|order)(?:\s+is)?\s*[:：-]?\s*', '', compact,
                     flags=re.IGNORECASE)
    compact = re.sub(r'\bclip\s*', '', compact, flags=re.IGNORECASE)
    compact = re.sub(r'\b(?:then|followed\s+by|to)\b', ',', compact,
                     flags=re.IGNORECASE)
    compact = compact.replace('→', ',').replace('>', ',')
    if not re.fullmatch(r'\s*[a-c]\s*(?:,|;|-|\s)\s*[a-c]\s*(?:,|;|-|\s)\s*[a-c]\s*',
                        compact, flags=re.IGNORECASE):
        return None, False
    answer = [x.upper() for x in re.findall(r'[a-c]', compact, flags=re.IGNORECASE)]
    return (answer, False) if sorted(answer) == list('ABC') else (None, False)


def _fallback_answer(text, row):
    candidate = text.strip()
    fenced = _strip_complete_fence(candidate)
    if fenced is not None:
        candidate = fenced
    elif candidate.startswith('```'):
        return None, False, True, 'incomplete_fence'

    parsed = None
    try:
        parsed = _loads_no_duplicates(candidate)
    except (ValueError, TypeError, json.JSONDecodeError):
        if candidate.startswith(('{', '[')):
            # Salvage exactly one terminal answer field from a closed object.
            # This covers an invalid/unescaped think string, but never an
            # unfinished object or duplicate/ambiguous answer fields.
            if not (candidate.startswith('{') and candidate.endswith('}')):
                return None, False, False, 'malformed_json'
            if row['task_id'] == 'A2':
                fields = re.findall(r'["\']answer["\']\s*:\s*["\']([^"\']+)["\']\s*}',
                                    candidate, flags=re.IGNORECASE)
                parsed = {'answer': fields[0]} if len(fields) == 1 else None
            else:
                fields = re.findall(r'["\']answer["\']\s*:\s*(\[[^\]]*\])\s*}',
                                    candidate, flags=re.IGNORECASE)
                try:
                    parsed = {'answer': json.loads(fields[0].replace("'", '"'))} if len(fields) == 1 else None
                except json.JSONDecodeError:
                    parsed = None
            if parsed is None:
                return None, False, len(fields) > 1, 'malformed_json'

    value = parsed.get('answer') if isinstance(parsed, dict) and 'answer' in parsed else parsed
    if parsed is None:
        # Never mine an essay for a convenient option letter.
        lines = [line.strip() for line in candidate.splitlines() if line.strip()]
        if len(candidate) > 120 or len(lines) > 2:
            return None, False, True, 'free_form_ambiguous'
        value = lines[-1] if lines else ''
        labels = {x.upper() for x in re.findall(r'(?<![A-Za-z])[A-E](?![A-Za-z])', value,
                                                flags=re.IGNORECASE)}
        if row['task_id'] == 'A2' and len(labels) > 1:
            return None, False, True, 'multiple_answers'
        if row['task_id'] == 'P1' and len(re.findall(
                r'(?<![A-Za-z])[A-C](?![A-Za-z])', value, flags=re.IGNORECASE)) > 3:
            return None, False, True, 'multiple_orders'

    if row['task_id'] == 'A2':
        answer, ambiguous = _normalise_a2(value)
    elif row['task_id'] == 'P1':
        answer, ambiguous = _normalise_p1(value)
    else:
        raise ValueError('Unsupported task')
    return answer, answer is not None, ambiguous, 'fallback'


def judge_answer(text, row, *, finish_reason=None, truncated=False,
                 output_tokens=None, max_tokens=None):
    """Length-aware strict-first judge with a conservative semantic fallback."""
    strict = evaluate_answer(text, row)
    hit_limit = generation_hit_limit(finish_reason=finish_reason, truncated=truncated,
                                     output_tokens=output_tokens, max_tokens=max_tokens)
    result = {
        'format_valid': strict['format_valid'],
        'content_exact_before_length': strict['exact_match'],
        'exact_match': False if hit_limit else strict['exact_match'],
        'semantic_match': False, 'accepted_match': False, 'fallback_applied': False,
        'hard_negative_length': hit_limit,
        'match_mode': 'length_hard_negative' if hit_limit else 'none',
        'review_required': False, 'normalized_answer': None, 'reward': 0.0,
    }
    if hit_limit:
        return result
    if strict['format_valid']:
        result.update(semantic_match=strict['exact_match'], accepted_match=strict['exact_match'],
                      match_mode='strict_correct' if strict['exact_match'] else 'strict_wrong',
                      reward=STRICT_REWARD if strict['exact_match'] else 0.0)
        return result
    answer, applied, ambiguous, mode = _fallback_answer(text, row)
    semantic = applied and answer == ground_truth(row)
    result.update(normalized_answer=answer, fallback_applied=applied,
                  semantic_match=semantic, accepted_match=semantic,
                  match_mode=('fallback_correct' if semantic else
                              'fallback_wrong' if applied else mode),
                  review_required=ambiguous,
                  reward=FALLBACK_REWARD if semantic else 0.0)
    return result


def sft_target(row):
    # The export contains no expert rationale. Empty think avoids invented labels.
    return json.dumps({'think': '', 'answer': ground_truth(row)}, separators=(',', ':'))


def prompt_parts(row):
    """Replace each filename with neutral clip/frame labels plus an image slot."""
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
    assert judge_answer(sft_target(a2), a2)['reward'] == STRICT_REWARD
    assert judge_answer('{"think":"","answer":"a"}', a2)['reward'] == FALLBACK_REWARD
    assert judge_answer('B, C, A', p1)['reward'] == FALLBACK_REWARD
    assert judge_answer(sft_target(a2), a2, finish_reason='length')['reward'] == 0
    print('TASK_CONTRACT_CHECKS_PASS')
