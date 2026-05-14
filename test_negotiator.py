import gc
import os
import argparse # 変更箇所
from multiprocessing import Pool
from itertools import product
import csv

import sao
from negmas import UtilityFunction, Issue
from negmas import load_genius_domain_from_folder # 変更箇所
from sao.my_sao import MySAOMechanism
from sao.my_negotiators import *
from envs.rl_negotiator import TestRLNegotiator
from matplotlib import pyplot as plt

ISSUE_NAMES = [
    'Laptop',
    'ItexvsCypress',
    'IS_BT_Acquisition',
    'Grocery',
    'thompson',
    'Car',
    'EnergySmall_A'
]
AGENT_LIST = [
    'Boulware',
    'Linear',
    'Conceder',
    'TitForTat1',
    'TitForTat2',
    "AgentK",
    "HardHeaded",
    "Atlas3",
    "AgentGG",
]

def a(x):
    return 'T' if x else 'F'


def run_session_trained(path, save_path, opponent, domain, util1, util2, util3, det, noise):
    session = MySAOMechanism(issues=domain, n_steps=80, avoid_ultimatum=False)
    opponent0 = get_opponent(opponent[0], agent_number=0, add_noise=noise) # 変更箇所
    opponent1 = get_opponent(opponent[1], agent_number=1, add_noise=noise) # 変更箇所
    my_agent = TestRLNegotiator(
        domain,
        path,
        deterministic=det,
        mode='issue',
        opponents=[opponent0.name, opponent1.name],
    )

    # 本実験では先攻・後攻の想定を考慮する必要がない
    session.add(my_agent, ufun=util1)
    session.add(opponent0, ufun=util2) # 変更箇所
    session.add(opponent1, ufun=util3) # 変更箇所

    result = session.run()

    # 結果を描画
    if PLOT:
        my_agent.name = "Our Agent"
        session.plot(path=save_path + os.path.basename(path).rsplit('.', maxsplit=1)[0] + f'-d{a(det)}-n{a(noise)}.png')
        # plt.show()
        plt.clf()
        plt.close()

    session.reset()
    del my_agent, session, opponent0, opponent1, util1._ami, util2._ami, util3._ami # 変更箇所
    gc.collect()

    if result['agreement'] is not None:
        my_util, opp_util1, opp_util2 = util1(result['agreement']), util2(result['agreement']), util3(result['agreement']) # 変更箇所
    else:
        my_util, opp_util1, opp_util2 = 0, 0, 0 # 変更箇所

    return [
        my_util,
        opp_util1, # 変更箇所
        opp_util2, # 変更箇所
        my_util + opp_util1 + opp_util2, # 変更箇所
        my_util * opp_util1 * opp_util2, # 変更箇所
        result['agreement'],
        result['step'], 
    ]
    # return {
    #     'my_util': my_util,
    #     'opp_util': opp_util,
    #     'social': my_util + opp_util,
    #     'nash': my_util * opp_util,
    #     'agreement': result['agreement'],
    #     'step': result['step']
    # }


def test_trained(config):
    issue, agent, det, noise, save_path = config
    
    results = [['my_util', 'opp_util1', 'opp_util2', 'social', 'nash', 'agreement', 'step']] # 変更箇所

    # ドメイン設定
    domain, _ = Issue.from_genius('./domain/' + issue + '/domain.xml')
    # !!!ここさえ変更すればOK!!!
    scenario_number1 = 0
    scenario_number2 = 1
    scenario_number3 = 2
    util1, _ = UtilityFunction.from_genius(f'./domain/{issue}/utility{scenario_number1+1}.xml')
    util2, _ = UtilityFunction.from_genius(f'./domain/{issue}/utility{scenario_number2+1}.xml')
    util3, _ = UtilityFunction.from_genius(f'./domain/{issue}/utility{scenario_number3+1}.xml')

    model_path = resolve_model_path(issue, agent)
    for _ in range(1 if PLOT else EPISODES):
        results.append(run_session_trained(model_path, save_path, agent, domain, util1, util2, util3, det, noise))

    if not PLOT:
        with open(f'{save_path}{issue}-{agent[0]}-{agent[1]}-d{a(det)}-n{a(noise)}.tsv', 'w') as f: # 変更箇所
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(results)


def resolve_model_path(issue, agent):
    load_path = LOAD_PATH
    checkpoint_path = os.path.join(load_path, "checkpoint.pt") if os.path.isdir(load_path) else load_path
    if checkpoint_path.endswith(".pt") and os.path.exists(checkpoint_path):
        return checkpoint_path
    return f'{LOAD_PATH}{issue}-{agent[0]}-{agent[1]}-v0.zip'


def build_result_path(load_path, plot, agent, issue, det, noise):
    result_root = os.path.dirname(load_path) if load_path.endswith(".pt") else load_path
    return os.path.join(
        result_root,
        'img' if plot else 'csv',
        f'{agent[0]}-{agent[1]}',
        issue,
        f'det={det}_noise={noise}',
    ) + os.sep


def get_opponent(opponent, agent_number=None, add_noise=False):
    suffix = "" if agent_number is None else str(agent_number)
    if opponent == 'Boulware':
        opponent = TimeBasedNegotiator(name='Boulware' + suffix, aspiration_type=10.0, add_noise=add_noise)
    elif opponent == 'Linear':
        opponent = TimeBasedNegotiator(name='Linear' + suffix, aspiration_type=1.0, add_noise=add_noise)
    elif opponent == 'Conceder':
        opponent = TimeBasedNegotiator(name='Conceder' + suffix, aspiration_type=0.2, add_noise=add_noise)
    elif opponent == 'TitForTat1':
        opponent = AverageTitForTatNegotiator(name='TitForTat1', gamma=1, add_noise=add_noise)
    elif opponent == 'TitForTat2':
        opponent = AverageTitForTatNegotiator(name='TitForTat2', gamma=2, add_noise=add_noise)
    elif opponent == 'AgentK':
        opponent = AgentK(add_noise=add_noise)
    elif opponent == 'HardHeaded':
        opponent = HardHeaded(add_noise=add_noise)
    elif opponent == 'CUHKAgent':
        opponent = CUHKAgent(add_noise=add_noise)
    elif opponent == 'Atlas3':
        opponent = Atlas3(add_noise=add_noise)
    elif opponent == 'AgentGG':
        opponent = AgentGG(add_noise=add_noise)
    else:
        opponent = TimeBasedNegotiator(name='Linear', aspiration_type=1.0, add_noise=add_noise)
    return opponent


def main_trained():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', '-a', nargs='*', required=True, type=str)
    parser.add_argument('--issues', '-i', nargs='*', required=True, type=str)
    parser.add_argument('--model', '-m', required=True, type=str)
    parser.add_argument('--scale', '-s', type=str, default='small')
    parser.add_argument('--plot', '-p', action='store_true')
    parser.add_argument('--is_first', '-if', action='store_true')
    parser.add_argument('--episodes', '-e', type=int, default=100)
    args = parser.parse_args()
    print(args)
    
    agents = args.agents
    issues = args.issues
    model_path = args.model
    scale = args.scale
    plot = args.plot
    is_first = args.is_first
    episodes = args.episodes

    global LOAD_PATH
    LOAD_PATH = model_path

    global PLOT
    PLOT = plot

    global EPISODES
    EPISODES = episodes
    
    if isinstance(issues, str):
        issues = [issues]
    if isinstance(agents, str):
        agents = [agents]

    p = Pool(len(agents))
    agent = [None, None] # 変更箇所
            
# 変更箇所
    for i in range(len(agents)):
        agent[0] = agents[i]
        for j in range(i, len(agents)):
            agent[1] = agents[j]
            for issue in issues:
                for det, noise in product([False], [False]):
                    save_path = build_result_path(LOAD_PATH, PLOT, agent, issue, det, noise)
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)
                        
                    p.map(test_trained, [(issue, agent, det, noise, save_path)])


if __name__ == '__main__':
    main_trained()
