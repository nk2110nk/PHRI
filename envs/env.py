import io
import sys

from .rl_negotiator import *
from .observer import *
from .domain_loader import load_genius_domain
from sao.my_sao import MySAOMechanism
from sao.my_negotiators import *

PENALTY = -1.


class NaiveEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, domain: str = 'party', opponent: list = ['Boulware','Boulware'], is_first: bool = False, test: bool = False):
        super().__init__()
        self.test = test
        
        # ドメイン設定
        # !!!ここさえ変更すればOK!!
        scenario_number1 = 0
        scenario_number2 = 1
        scenario_number3 = 2
        self.domain, utilities = load_genius_domain(
            domain,
            (scenario_number1, scenario_number2, scenario_number3),
        )
        self.util1, self.util2, self.util3 = utilities

        self.my_agent: Optional[RLNegotiator] = None
        self.session: Optional[MySAOMechanism] = None

        # 設定読み込み
        self.opponent = opponent
        self.is_first = is_first
        
        self.my_util = self.util1
        self.opp_util1 = self.util2
        self.opp_util2 = self.util3

        # 強化学習関連
        self.state = None
        self.action = None
        self.observation = None
        self.observation_opponent = None

        # self.observer = OneHotObserve2n(self.domain, 20)
        self.observer = OnehotObserve2nT(self.domain, 6) # 変更箇所
        # self.observer = OpponentObserve1(self.domain)
        self.all_bids = self.get_all_bids()
        self.observation_space = self.observer.observation_space
        self.action_space = gym.spaces.Discrete(len(self.all_bids))
        self.reward_range = [PENALTY, 1.0]
        self.seed()

    def reset(self):
        # セッション，エージェントの作成
        del self.my_util._ami
        del self.opp_util1._ami
        del self.opp_util2._ami
        if self.session is not None:
            self.session.reset()
        # self.session.__init__(issues=self.domain, n_steps=80, avoid_ultimatum=False)
        self.session = MySAOMechanism(issues=self.domain, n_steps=80, avoid_ultimatum=False)
        self.my_agent = RLNegotiator()
        
        # 対戦相手の追加
        opponent = []
        # 変更箇所
        num_opponent_negotiator = len(self.opponent)
        for i in range(num_opponent_negotiator):
            self.agent_number = i
            opponent.append(self.get_opponent(add_noise=True))

        # セッションにエージェントの追加
        self.session.add(self.my_agent, ufun=self.my_util)
        self.session.add(opponent[0], ufun=self.opp_util1) # 変更箇所
        self.session.add(opponent[1], ufun=self.opp_util2) # 変更箇所
        # 自分から提案
        self.state = None

        self.observation_opponent = [opp.name for opp in opponent] # 変更箇所
        self.observer.reset()
        self.observation = self.observer(self.state, self.observation_opponent) # 変更箇所
        return self.observation

    def step(self, action: int):
        self.action = self.all_bids[action]
        self.my_agent.set_next_bid(self.action)
        
        self.state = self.session.step().__dict__
        # 状態を更新 変更箇所
        self.observation = self.observer(self.state, self.observation_opponent) # 変更箇所
        if self.state['agreement'] is not None:  # 合意していたら
            return self.observation, self.get_reward(), True, {}
        if self.state['timedout'] or self.state['broken']:
            return self.observation, self.get_reward(), True, {}
        return self.observation, self.get_reward(), False, {}

    def render(self, mode='human', close=False):
        outfile = io.StringIO() if mode == 'ansi' else sys.stdout
        if not self.state['running']:
            # outfile.write(
            #     '\nsteps:' + str(self.state['step']) + ', util:' +
            #     str(self.get_reward()) + '\n'
            # )
            self.session.plot()
            # self.make_log()
        return outfile

    def make_log(self):
        print('step,message,proposer,' + ','.join([issue.name for issue in self.domain]))
        for state in self.session.history:
            if state.agreement is None:
                print(f'{state.step},offer,{state.current_proposer},{self.bid2str(state.current_offer)}')
            else:
                print(f'{state.step},accept,{state.current_proposer},{self.bid2str(state.agreement)}')

    def bid2str(self, bid, onehot=False):
        if onehot:
            return ','.join([''.join(['1' if bid[issue.name] == value else '0' for value in issue.values]) for issue in
                             self.domain])
        else:
            return ','.join([bid[issue.name] for issue in self.domain])

    # 変更箇所
    def get_opponent(self, add_noise=False):
        if self.opponent[self.agent_number] == 'Boulware':
            opponent = TimeBasedNegotiator(name = 'Boulware{}'.format(self.agent_number), aspiration_type=10.0, add_noise=add_noise)
        elif self.opponent[self.agent_number] == 'Linear':
            opponent = TimeBasedNegotiator(name='Linear{}'.format(self.agent_number), aspiration_type=1.0, add_noise=add_noise)
        elif self.opponent[self.agent_number] == 'Conceder':
            opponent = TimeBasedNegotiator(name='Conceder{}'.format(self.agent_number), aspiration_type=0.2, add_noise=add_noise)
        elif self.opponent[self.agent_number] == 'TitForTat1':
            opponent = AverageTitForTatNegotiator(name='TitForTat1', gamma=1, add_noise=add_noise)
        elif self.opponent[self.agent_number] == 'TitForTat2':
            opponent = AverageTitForTatNegotiator(name='TitForTat2', gamma=2, add_noise=add_noise)
        elif self.opponent[self.agent_number] == 'AgentK':
            opponent = AgentK(add_noise=add_noise)
        elif self.opponent[self.agent_number] == 'HardHeaded':
            opponent = HardHeaded(add_noise=add_noise)
        #elif self.opponent[self.agent_number] == 'CUHKAgent':
            #opponent = CUHKAgent(add_noise=add_noise)
        elif self.opponent[self.agent_number] == 'Atlas3':
            opponent = Atlas3(add_noise=add_noise)
        elif self.opponent[self.agent_number] == 'AgentGG':
            opponent = AgentGG(add_noise=add_noise)
        else:
            opponent = TimeBasedNegotiator(name='Linear', aspiration_type=1.0, add_noise=add_noise)
        return opponent

    def close(self):
        del self.domain
        del self.util1
        del self.util2
        del self.util3 # 変更箇所
        del self.my_util
        del self.opp_util1 # 変更箇所
        del self.opp_util2 # 変更箇所
        del self.my_agent
        del self.all_bids
        del self.session

    def seed(self, seed=None):
        pass

    def get_reward(self):
        if self.state['timedout'] or self.state['broken']:
            if not self.test:
                return PENALTY
            else:
                return 0
        elif self.state['agreement'] is not None:
            return self.my_util(self.state['agreement'])
        else:
            return 0

    def get_all_bids(self):
        session = MySAOMechanism(issues=self.domain, n_steps=80, avoid_ultimatum=False)
        agent = RLNegotiator()
        session.add(agent, ufun=self.util1)
        return agent.all_bids


class AOPEnv(NaiveEnv):
    def __init__(self, domain='party', opponent=['Boulware','Boulware'], is_first=False, test=False):
        super().__init__(domain, opponent, is_first, test)
        self.action_space = gym.spaces.Discrete(len(self.all_bids) + 1)

    def step(self, action: int):
        if action == len(self.all_bids):
            if self.state is None:
                self.state = {'broken': True, 'timedout': False}
            else:
                self.state = {k: self.state['current_offer'] if k == 'agreement' else False if k == 'running' else v for
                              (k, v) in self.state.items()}
                self.observation = self.observer(self.state, self.observation_opponent) # 変更箇所
            return self.observation, self.get_reward(), True, {}
        else:
            return super().step(action)


class DenseEnv(AOPEnv):
    def get_reward(self):
        if self.state['timedout'] or self.state['broken']:
            if not self.test:
                return PENALTY
            else:
                return 0
        elif self.state['agreement'] is not None:
            return self.my_util(self.state['agreement'])
        else:   # ここを変える
            return self.my_util(self.state['current_offer']) / 100


class IssueActionEnv(AOPEnv):
    is_acceptable = True

    def __init__(self, domain='party', opponent=['Boulware','Boulware'], is_first=False, test=False):
        super().__init__(domain, opponent, is_first, test)
        if self.is_acceptable:
            # [issue, (accept, (end), reject)]
            self.action_space = gym.spaces.MultiDiscrete([len(i.values) for i in self.domain] + [2])
        else:
            self.action_space = gym.spaces.MultiDiscrete([len(i.values) for i in self.domain])

    def step(self, action: list):
        if self.is_acceptable:
            if action[-1] == 0:    # accept
                if self.state is None:
                    self.state = {'broken': True, 'timedout': False}
                else:
                    self.state = {k: self.state['current_offer'] if k == 'agreement' else False if k == 'running' else v for
                                  (k, v) in self.state.items()}
                    self.observation = self.observer(self.state, self.observation_opponent) # 変更箇所
                return self.observation, self.get_reward(), True, {}
            # elif action[-1] == 1:  # end
            #     self.state = {'broken': True, 'timedout': False}
            #     return self.observation, self.get_reward(), True, {}
            else:                       # reject
                self.action = {i.name: i.values[v] for i, v in zip(self.domain, action)}
                self.my_agent.set_next_bid(self.action)
                self.state = self.session.step().__dict__
                # 状態を更新 変更箇所
                self.observation = self.observer(self.state, self.observation_opponent) # 変更箇所
                if self.state['agreement'] is not None:  # 合意していたら
                    return self.observation, self.get_reward(), True, {}
                if self.state['timedout'] or self.state['broken']:
                    return self.observation, self.get_reward(), True, {}
                return self.observation, self.get_reward(), False, {}
        else:
            self.action = {i.name: i.values[v] for i, v in zip(self.domain, action)}
            self.my_agent.set_next_bid(self.action)
            self.state = self.session.step().__dict__
            # 状態を更新 変更箇所
            self.observation = self.observer(self.state, self.observation_opponent) # 変更箇所
            if self.state['agreement'] is not None:  # 合意していたら
                return self.observation, self.get_reward(), True, {}
            if self.state['timedout'] or self.state['broken']:
                return self.observation, self.get_reward(), True, {}
            return self.observation, self.get_reward(), False, {}


# class RLBOAEnv(NaiveEnv):
#     def __init__(self, domain='party', opponent='Boulware', is_first=False, test=False, n_actions=10):
#         super().__init__(domain, opponent, is_first, test)
#         self.n_actions = 10
#         self.observer = RLBOAObserve(self.domain, self.my_util)
#         self.observation_space = self.observer.observation_space
#         self.action_space = gym.spaces.Discrete(n_actions)

#     def reset(self):
#         # セッション，エージェントの作成
#         del self.my_util._ami
#         del self.opp_util._ami
#         if self.session is not None:
#             del self.my_agent.om
#             self.session.reset()

#         # self.session.__init__(issues=self.domain, n_steps=80, avoid_ultimatum=False)
#         self.session = MySAOMechanism(issues=self.domain, n_steps=80, avoid_ultimatum=False)
#         self.my_agent = RLBOANegotiator()
#         opponent = self.get_opponent()

#         # セッションにエージェントの追加
#         if self.is_first:
#             self.session.add(self.my_agent, ufun=self.my_util)
#             self.session.add(opponent, ufun=self.opp_util)
#             self.my_agent.om = HardHeadedFrequencyModel(self.my_util)
#             self.state = None
#         else:
#             self.session.add(opponent, ufun=self.opp_util)
#             self.session.add(self.my_agent, ufun=self.my_util)
#             self.my_agent.om = HardHeadedFrequencyModel(self.my_util)
#             # 後攻だったら相手に1回提案させる
#             self.state = self.session.step().__dict__

#         self.observer.reset()
#         self.observation = self.observer(self.state)
#         return self.observation

#     def step(self, action: int):
#         self.action = action
#         self.my_agent.set_target(self.action)
#         for _ in range(2):
#             self.state = self.session.step().__dict__
#             # 状態を更新
#             self.observation = self.observer(self.state)
#             if self.state['agreement'] is not None:  # 合意していたら
#                 return self.observation, self.get_reward(), True, {}
#             if self.state['timedout'] or self.state['broken']:
#                 return self.observation, self.get_reward(), True, {}
#         return self.observation, self.get_reward(), False, {}
