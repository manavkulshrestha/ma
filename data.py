from numpy.linalg import norm
import numpy as np

from pathlib import Path

import re

TEMPLATES_DIR = Path('./mjcf/templates/')


def sample_nonoverlaping(existing, *, sample_range, size=2, thresh=0.1):
    """
    uniformly samples a size-dimensional point which is at least far away from all points in existing by thresh

    existing: a list of existing points
    low: lower bound for uniform sampling
    high: upper bound for uniform sampling
    size: the dimension of the sampled point
    thresh: at least how far new point must be from existing ones
    """
    low, high = sample_range
    if not existing:
        return np.random.uniform(low, high, size=size)
  
    while True:
        ret = np.random.uniform(low, high, size=size)
        if (norm(np.array(existing) - ret, axis=1) > thresh).all():
            return ret
        
def multiple_replace(s, replace_dict):
    """ replaces each occurance of a key in s with its value as they exist in replace_dict"""
    for old, new in replace_dict.items():
        s = re.sub(old, new, s)

    return s

def add_entity(env_template, existing_pos, *, e_nrange, env_bounds, entities_marker, etemplate_path, ename_fmt, posi_fmt):
    with open(etemplate_path, 'r') as f:
        entity_template = f.read()

    entity_strs = []
    num_entities = np.random.randint(*e_nrange)
    for i in range(num_entities):
        posi = ' '.join(map(str, sample_nonoverlaping(existing_pos, sample_range=env_bounds)))
        entityi = multiple_replace(entity_template, {
            r'<(BODY|JOINT|SITE)>': ename_fmt % i,
            '<POS>': posi_fmt % posi
        })
        entity_strs.append(entityi)

    return env_template.replace(entities_marker, '\n\n'.join(entity_strs)), existing_pos, num_entities
    
def add_robots(env_template, existing_pos, *, nrange, env_bounds, simple=False):
    template_name = f'robot_template{"_simple" if simple else ""}.xml'
    position_format = f'%s {0.05 if simple else 0.01}'
    return add_entity(env_template, existing_pos,
                      e_nrange=nrange, env_bounds=env_bounds,
                      entities_marker='<!-- robots -->', etemplate_path=TEMPLATES_DIR/template_name,
                      ename_fmt='robot%d', posi_fmt=position_format)

def add_humans(env_template, existing_pos, *, nrange, env_bounds):
    return add_entity(env_template, existing_pos,
                      e_nrange=nrange, env_bounds=env_bounds,
                      entities_marker='<!-- humans -->', etemplate_path=TEMPLATES_DIR/'human_template.xml',
                      ename_fmt='human%d', posi_fmt='%s 0.05')

def add_goals(env_template, existing_pos, *, nrange, env_bounds):
    return add_entity(env_template, existing_pos,
                      e_nrange=nrange, env_bounds=env_bounds,
                      entities_marker='<!-- goals -->', etemplate_path=TEMPLATES_DIR/'goal_template.xml',
                      ename_fmt='goal%d', posi_fmt='%s 0')

def prepare_episode(template_path, *, r_nrange, h_nrange, env_bounds, episode_path, simple_robots=False):
    """ prepares an episode given ranges for random parameters """
    with open(template_path, 'r') as f:
        template = f.read()

    existing_pos = []
    template, existing_pos, num_robots = add_robots(template, existing_pos, nrange=r_nrange, env_bounds=env_bounds, simple=simple_robots)
    template, existing_pos, num_humans = add_humans(template, existing_pos, nrange=h_nrange, env_bounds=env_bounds)
    # template, existing_pos = add_goals(template, existing_pos, nrange=h_nrange, env_bounds=env_bounds)

    with open(episode_path, 'w+') as f:
        f.write(template)

    return template, num_robots, num_humans
