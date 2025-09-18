import numpy as np
import pandas as pd

class QLearningTable:
	def __init__(self, actions, learning_rate=0.01, reward_decay=0.5, e_greedy = 0.7, e_greedy_increment=0.9,  q_table=None):
		self.actions = actions
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon_explor = e_greedy
		self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
		self.epsilon_max = e_greedy_increment
		if q_table is None:
			self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
		else:
			self.q_table = q_table

	def choose_action(self, observation, note):
		seed_value = 1234  # 你可以选择任何整数作为种子

		np.random.seed(seed_value)
		self.check_state_exist(observation)
		if note ==0:
			epsilon = self.epsilon_explor
		else:
			epsilon = self.epsilon_max

		if np.random.uniform() < epsilon:
			state_action = self.q_table.loc[observation,:]#######获取当前状态下所有动作的 Q 值
			action = np.random.choice(state_action[state_action==np.max(state_action)].index)####可能有几个动作都是最大q，随机从中凑一个
		else:
			action = np.random.choice(self.actions)
		return action

	def learn(self, s, a, r, s_):
		self.check_state_exist(s_)
		#print("done")
		q_predict = self.q_table.loc[s, a]####获取当前状态下a动作的q值
		if s_ != 'terminal':

			q_target = r + self.gamma*self.q_table.loc[s_, :].max()####未结束
		else:
			q_target = r
		self.q_table.loc[s, a] += self.lr*(q_target-q_predict)

	def check_state_exist(self, state):
		if state not in self.q_table.index:
			new_series = pd.Series([0]*len(self.actions), index=self.q_table.columns, name=state)
			self.q_table = pd.concat([self.q_table, new_series.to_frame().T], ignore_index=False)