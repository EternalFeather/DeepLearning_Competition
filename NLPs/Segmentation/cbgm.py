from collections import defaultdict
import pickle
from functools import reduce
from math import log
import heapq


class Cbgm(object):
	def __init__(self):
		self.l1 = 0.0
		self.l2 = 0.0
		self.l3 = 0.0
		self.status = {'b', 'm', 'e', 's'}
		self.uni = defaultdict(int)
		self.bi = defaultdict(int)
		self.tri = defaultdict(int)

	def train(self, data):
		for sentence in data:
			current = [('', 'BOS'), ('', 'BOS')]
			self.bi[(('', 'BOS'), ('', 'BOS'))] += 1
			self.uni[('', 'BOS')] += 2
			for word, tag in sentence:
				current.append((word, tag))
				self.uni[(word, tag)] += 1
				self.bi[tuple(current[1:])] += 1
				self.tri[tuple(current)] += 1
				current.pop(0)
		tl1 = 0.0
		tl2 = 0.0
		tl3 = 0.0
		for current in self.tri.keys():
			c3 = self.cbgm_div(self.tri[current] - 1, self.bi[current[:2]] - 1)
			c2 = self.cbgm_div(self.bi[current[1:]] - 1, self.uni[current[1]] - 1)
			c1 = self.cbgm_div(self.uni[current[2]] - 1, self.cbgm_total(self.uni) - 1)
			if c3 >= c1 and c3 >= c2:
				tl3 += self.tri[current]
			elif c2 >= c1 and c2 >= c3:
				tl2 += self.tri[current]
			elif c1 >= c3 and c1 >= c2:
				tl1 += self.tri[current]
		self.l1 = float(tl1) / (tl1 + tl2 + tl3)
		self.l2 = float(tl2) / (tl1 + tl2 + tl3)
		self.l3 = float(tl3) / (tl1 + tl2 + tl3)

		self.save(self.uni, './Segmentation/checkpoints/cbgm/uni.pkl')
		self.save(self.bi, './Segmentation/checkpoints/cbgm/bi.pkl')
		self.save(self.tri, './Segmentation/checkpoints/cbgm/tri.pkl')
		self.save(self.l1, './Segmentation/checkpoints/cbgm/l1.pkl')
		self.save(self.l2, './Segmentation/checkpoints/cbgm/l2.pkl')
		self.save(self.l3, './Segmentation/checkpoints/cbgm/l3.pkl')

	def cbgm_div(self, v1, v2):
		if v2 == 0:
			return 0
		return float(v1) / v2

	def cbgm_total(self, d):
		return reduce(lambda x, y: x + y, map(lambda x: x[1], d.items()))

	def save(self, d, fname):
		pickle.dump(d, open(fname, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

	def restore(self):
		self.uni = pickle.load(open('./Segmentation/checkpoints/cbgm/uni.pkl', 'rb'))
		self.bi = pickle.load(open('./Segmentation/checkpoints/cbgm/bi.pkl', 'rb'))
		self.tri = pickle.load(open('./Segmentation/checkpoints/cbgm/tri.pkl', 'rb'))
		self.l1 = pickle.load(open('./Segmentation/checkpoints/cbgm/l1.pkl', 'rb'))
		self.l2 = pickle.load(open('./Segmentation/checkpoints/cbgm/l2.pkl', 'rb'))
		self.l3 = pickle.load(open('./Segmentation/checkpoints/cbgm/l3.pkl', 'rb'))

	def log_prob(self, s1, s2, s3):
		uni = self.l1 * self.cbgm_div(self.uni[s3], self.cbgm_total(self.uni))
		bi = self.l2 * self.cbgm_div(self.bi[(s2, s3)], self.uni[s2])
		tri = self.l3 * self.cbgm_div(self.tri[(s1, s2, s3)], self.bi[(s1, s2)])
		return log(uni + bi + tri + 1)

	def cut(self, sentence):
		current = [((('', 'BOS'), ('', 'BOS')), 0.0, [])]
		for w in sentence:
			stage = {}
			flag = True
			for t in self.status:
				if t in self.uni.keys():
					flag = False
					break
			if flag:
				for t in self.status:
					for pre in current:
						stage[(pre[0][1], (w, t))] = (pre[1], pre[2] + [t])
				current = list(map(lambda x: (x[0], x[1][0], x[1][1]), stage.items()))
				continue
			for t in self.status:
				for pre in current:
					p = pre[1] + self.log_prob(pre[0][0], pre[0][1], (w, t))
					if (pre[0][1], (w, t)) not in stage or p > stage[(pre[0][1], (w, t))][0]:
						stage[(pre[0][1], (w, t))] = (p, pre[2] + [t])
			current = list(map(lambda x: (x[0], x[1][0], x[1][1]), stage.items()))
		current = heapq.nlargest(1, current, key=lambda x: x[1])
		return zip(current, current[2])
