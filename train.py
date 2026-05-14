import dill
dill.extend(False)
import gym
import sao
import sys
import os
import argparse # 変更箇所
from multiprocessing import Pool
from datetime import datetime
from gym import register

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from datetime import datetime # 変更箇所
# from itertools import combinations_with_replacement # 変更箇所
from ppo_scratch import DEFAULT_GENERAL_DOMAIN, PPO as MiPNScratchPPO
dill.extend(True)

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
ENV_LIST = [
    ('IssueActionEnv-{}-{}-{}-v0', 'envs.env:IssueActionEnv'),
    ('AOPEnv-{}-{}-{}-v0', 'envs.env:AOPEnv'),
]


def register_neg_env(issue, agents, env):
    if len(agents) == 1:
        agents = [agents[0], agents[0]]
    env_name = env[0].format(issue, agents[0], agents[1])
    register(
        id=env_name,
        entry_point=env[1],
        kwargs={'domain': issue, 'opponent': agents, 'is_first': True},
    )
    return env_name


def run_rl(args):
    issue, agents, e_tuple, save_path = args
    env_name = register_neg_env(issue, agents, e_tuple)
    # f_name = env_name.split('-', maxsplit=1)[1]
    f_name = "checkpoint" # 変更箇所
    env = make_vec_env(env_name, n_envs=4)

    model = PPO("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log=save_path)
    model.learn(total_timesteps=500000, tb_log_name=f_name) # もしtimestepsを変更する場合は、ここを変更
    model.save(save_path + f_name)

    # Use a separate environement for evaluation
    eval_env = gym.make(env_name)
    eval_env.test = True
    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    with open(save_path + "result.csv", "a") as f:
        f.write("{},{},{},{},{}\n".format(*env_name.split('-')[1:4], mean_reward, std_reward)) # 変更箇所

    env.close()
    eval_env.close()
    del model


def build_save_path(issues, agents, current_time, save_path):
    if save_path != './results/':
        return save_path if save_path.endswith('/') else save_path + '/'
    return "./results/{}_{}/{}-TA/".format('-'.join(issues), '-'.join(agents), current_time)


def main_issue(
    agents,
    issues,
    total_timesteps,
    n_envs,
    n_rollout_steps,
    random_train=False,
    general_domain=DEFAULT_GENERAL_DOMAIN,
):
    save_path = SAVE_PATH + 'MiPN_Negotiator/'
    os.makedirs(save_path, exist_ok=True)
    with open(save_path + "result.csv", "w") as f:
        f.write("domain,opponents,general_domain,total_timesteps,n_envs,n_rollout_steps,checkpoint\n")

    model = MiPNScratchPPO(
        issue=issues,
        agents=agents,
        n_envs=n_envs,
        n_rollout_steps=n_rollout_steps,
        device="auto",
        random_train=random_train,
        general_domain=general_domain,
    )
    model.train(total_timesteps=total_timesteps, save_path=save_path)

    with open(save_path + "result.csv", "a") as f:
        f.write("{},{},{},{},{},{},{}\n".format(
            "-".join(issues),
            "-".join(agents),
            general_domain,
            total_timesteps,
            n_envs,
            n_rollout_steps,
            save_path + "checkpoint.pt",
        ))

    # p = Pool(len(agents))
    # pairs = list(combinations_with_replacement(agents, 2)) # 変更箇所
    
    # for issue in issues:
    #     p.map(run_rl, [(issue, agent_set, ENV_LIST[0], save_path) for agent_set in pairs]) # 変更箇所


def main_aop(agents, issues):
    save_path = SAVE_PATH + 'VeNAS/'
    os.makedirs(save_path)
    with open(save_path + "result.csv", "w") as f:
        f.write("domain,opponent,mean,std\n")
        
    run_rl((issues[0], agents, ENV_LIST[1], save_path)) # ここはexpertならいいけど、のちに変更必須

    # p = Pool(len(agents))
    # pairs = list(combinations_with_replacement(agents, 2)) # 変更箇所
    
    # for issue in issues:
    #     p.map(run_rl, [(issue, agent_set, ENV_LIST[1], save_path) for agent_set in pairs]) # 変更箇所


def main():
    # 変更箇所
    # 時間記録
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # IssueとAgentを指定して実行 -> --agents, --issue
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', '-a', required=True, nargs='*', type=str)
    parser.add_argument('--issue', '-i', required=True, nargs='*', type=str)
    parser.add_argument('--save_path', '-sp', type=str, default="./results/")
    parser.add_argument('--timesteps', '-t', type=int, default=500000)
    parser.add_argument('--n_envs', '-n', type=int, default=4)
    parser.add_argument('--n_rollout_steps', '-rs', type=int, default=2048)
    parser.add_argument('--skip_venas', action='store_true')
    parser.add_argument('--random_train', action='store_true')
    parser.add_argument('--general_domain', '-gd', type=str, default=DEFAULT_GENERAL_DOMAIN)
    args = parser.parse_args()
    agents = args.agents
    issue = args.issue
    save_path = args.save_path
    total_timesteps = args.timesteps
    n_envs = args.n_envs
    n_rollout_steps = args.n_rollout_steps
    random_train = args.random_train
    general_domain = args.general_domain
    #print(args)
    
    global SAVE_PATH
    SAVE_PATH = build_save_path(issue, agents, current_time, save_path)
    
    main_issue(
        agents,
        issue,
        total_timesteps,
        n_envs,
        n_rollout_steps,
        random_train=random_train,
        general_domain=general_domain,
    ) # MiPN_Negotiator
    if not args.skip_venas:
        main_aop(agents, issue) # VeNAS


if __name__ == '__main__':
    main()
