from pathlib import Path
import time
import numpy as np
import mujoco as mj
import mujoco.viewer as mjv

from agent import Human, Robot
from data import TEMPLATES_DIR, prepare_episode
from utility import save_pkl, time_label

EPISODES_PATH = Path('./mjcf/episodes/')
DATA_PATH = Path('./data/')

def run_episode(episode_path, episode_length=1000):
    mjcf, num_robots, num_humans = prepare_episode(TEMPLATES_DIR/'env_template.xml',
                                                    r_nrange=(3, 6), h_nrange=(3, 6), env_bounds=(-1, 1),
                                                    simple_robots=True,
                                                    episode_path=episode_path)
    m = mj.MjModel.from_xml_path(str(episode_path))
    d = mj.MjData(m)

    humans = [Human(f'human{i}', robots_num=num_robots, model=m, data=d) for i in range(num_humans)]
    robots = [Robot(f'robot{i}', model=m, data=d) for i in range(num_robots)]

    timeseries_data = []
    t = 0

    with mjv.launch_passive(m, d) as viewer:
        while viewer.is_running() and t < episode_length:
            # policy
            for h in humans:
                h.wander()
            actions = [r.herd() for r in robots]

            mj.mj_step(m, d)
            viewer.sync()

            # data collection:
            robots_data = [{'pos':r.pos, 'vel':r.vel, 'action':a, 'travelled': h.travelled} for r,a in zip(robots, actions)]
            humans_data = [{'pos':h.pos, 'vel':h.vel, 'travelled': h.travelled} for h in humans]

            timeseries_data.append({'t':t, 'robots':robots_data, 'humans':humans_data})
            t += 1

    return {
        'num_robots': num_robots, 
        'num_humans': num_humans,
        'episode_length': len(timeseries_data),
        'timesteps': timeseries_data
    }


def main():
    num_episodes = 10000
    run_name = time_label()
    run_dir = DATA_PATH/run_name
    eps_dir = EPISODES_PATH/run_name

    eps_dir.mkdir(parents=True, exist_ok=False)
    run_dir.mkdir(parents=True, exist_ok=False)

    for i in range(num_episodes):
        episode_name = time_label()
        episode_data = run_episode(eps_dir/episode_name, episode_length=2000)
        save_pkl(episode_data, run_dir/f'{episode_name}.pkl')


if __name__ == '__main__':
    main()