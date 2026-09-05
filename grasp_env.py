"""A one-action image-and-text environment for the pinned VAGEN-Lite branch."""
from vagen.envs.gym_image_env import GymImageEnv
from grasp_common import SYSTEM, load_rows, load_visual
from task_contract import evaluate_answer
import json
from pathlib import Path

class GraSPEnv(GymImageEnv):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.rows = load_rows(env_config['split'])
        self.row = None

    async def reset(self, seed):
        if not 0 <= int(seed) < len(self.rows):
            raise ValueError('Explicit row index out of range')
        self.row = self.rows[int(seed)]
        self.seed = int(seed)
        parts, images = load_visual(self.row)
        text = ''.join(p['text'] if p['type'] == 'text' else '<image>' for p in parts)
        return {'obs_str': text, 'multi_modal_input': {'<image>': images}}, {'success': False}

    async def system_prompt(self):
        return {'obs_str': SYSTEM, 'multi_modal_input': {}}

    async def step(self, action_str):
        result = evaluate_answer(action_str, self.row)
        record = {'seed':self.seed,'split':self.config['split'],'task_id':self.row['task_id'],
                  'response':action_str,**result}
        if self.config.get('rollout_log'):
            with Path(self.config['rollout_log']).open('a') as stream:
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(f"GRASP_ROLLOUT_DONE task={self.row['task_id']} seed={self.seed} reward={result['reward']}",flush=True)
        return {'obs_str': '', 'multi_modal_input': {}}, result['reward'], True, {
            'success': result['exact_match'], 'format_valid': result['format_valid']}

    async def close(self):
        self.row = None
