"""A one-action image-and-text environment for the pinned VAGEN-Lite branch."""
from vagen.envs.gym_image_env import GymImageEnv
from grasp_common import SYSTEM, load_rows, load_visual
from task_contract import judge_answer
import json
import inspect
from pathlib import Path


def install_generation_limit_bridge():
    """Expose VAGEN's token-limit metadata to the environment before reward."""
    from vagen.agent_loop.gym_agent_loop import GymAgentLoop
    original = GymAgentLoop._handle_env_state
    if getattr(original, '_grasp_length_bridge', False):
        return
    if not inspect.iscoroutinefunction(original):
        raise RuntimeError('Pinned VAGEN contract changed: _handle_env_state is not async')

    async def wrapped(self, agent_data, *args, **kwargs):
        response_ids = getattr(agent_data, 'response_ids', None)
        response_limit = getattr(agent_data, 'response_limit', None)
        env = getattr(agent_data, 'env', None)
        if env is None or response_ids is None or response_limit is None:
            raise RuntimeError('Pinned VAGEN contract changed: rollout length metadata missing')
        env.last_generation_output_tokens = len(response_ids)
        env.last_generation_max_tokens = int(response_limit)
        env.last_generation_hit_limit = len(response_ids) >= int(response_limit)
        return await original(self, agent_data, *args, **kwargs)

    wrapped._grasp_length_bridge = True
    wrapped._grasp_original = original
    GymAgentLoop._handle_env_state = wrapped


install_generation_limit_bridge()

class GraSPEnv(GymImageEnv):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.rows = load_rows(env_config['split'])
        self.row = None
        self.last_generation_hit_limit = False
        self.last_generation_output_tokens = None
        self.last_generation_max_tokens = None

    async def reset(self, seed):
        if not 0 <= int(seed) < len(self.rows):
            raise ValueError('Explicit row index out of range')
        self.row = self.rows[int(seed)]
        self.seed = int(seed)
        self.last_generation_hit_limit = False
        self.last_generation_output_tokens = None
        self.last_generation_max_tokens = None
        parts, images = load_visual(self.row)
        text = ''.join(p['text'] if p['type'] == 'text' else '<image>' for p in parts)
        return {'obs_str': text, 'multi_modal_input': {'<image>': images}}, {'success': False}

    async def system_prompt(self):
        return {'obs_str': SYSTEM, 'multi_modal_input': {}}

    async def step(self, action_str):
        result = judge_answer(action_str, self.row,
                              truncated=self.last_generation_hit_limit,
                              output_tokens=self.last_generation_output_tokens,
                              max_tokens=self.last_generation_max_tokens)
        record = {'seed':self.seed,'split':self.config['split'],'task_id':self.row['task_id'],
                  'response':action_str,**result}
        if self.config.get('rollout_log'):
            with Path(self.config['rollout_log']).open('a') as stream:
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(f"GRASP_ROLLOUT_DONE task={self.row['task_id']} seed={self.seed} reward={result['reward']}",flush=True)
        return {'obs_str': '', 'multi_modal_input': {}}, result['reward'], True, {
            'success': result['accepted_match'], 'format_valid': result['format_valid'],
            'match_mode': result['match_mode'],
            'hard_negative_length': result['hard_negative_length']}

    async def close(self):
        self.row = None
