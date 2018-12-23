from collections import defaultdict
from functools import reduce
from math import log
import pickle
import heapq


class AddOneProb(object):
	def __init__(self):
		self.d = {}

	def addone(self, key, value):
		if key not in self.d:
			self.d[key] = 1
		self.d[key] += value


class Tnt(object):
	def __init__(self, bean_search=1000):
		self.bean_search = bean_search
		self.l1 = 0.0
		self.l2 = 0.0
		self.l3 = 0.0
		self.status = set()		# 所有可能的tags
		self.wd = AddOneProb()		# tag->word 的对应数量
		self.eos = AddOneProb()		# bi-gram 结尾词数
		self.eosd = AddOneProb()		# uni-gram 结尾词数
		self.uni = defaultdict(int)		# uni-gram 词数
		self.bi = defaultdict(int)		# bi-gram 词数
		self.tri = defaultdict(int)		# tri-gram 词数
		self.word = {}		# 所有的word具有的tag种类
		self.trans = {}		#

	def train(self, data):
		for sentence in data:
			current = ['BOS', 'BOS']
			self.bi[('BOS', 'BOS')] += 1
			self.uni['BOS'] += 2
			for word, tag in sentence:
				current.append(tag)
				self.status.add(tag)
				self.wd.addone((tag, word), 1)
				self.eos.addone(tuple(current[1:]), 1)
				self.eosd.addone(tag, 1)
				self.uni[tag] += 1
				self.bi[tuple(current[1:])] += 1
				self.tri[tuple(current)] += 1
				temp_word = self.word.setdefault(word, set())
				temp_word.add(tag)
				current.pop(0)
			self.eos.addone((current[-1], 'EOS'), 1)
		tl1 = 0.0
		tl2 = 0.0
		tl3 = 0.0
		for current in self.tri.keys():
			c3 = self.tnt_div(self.tri[current] - 1, self.bi[current[:2]] - 1)
			c2 = self.tnt_div(self.bi[current[1:]] - 1, self.uni[current[1]] - 1)
			c1 = self.tnt_div(self.uni[current[2]] - 1, self.tnt_total(self.uni) - 1)
			if c3 >= c2 and c3 >= c1:
				tl3 += self.tri[current]
			elif c2 >= c3 and c2 >= c1:
				tl2 += self.tri[current]
			elif c1 >= c3 and c1 >= c2:
				tl1 += self.tri[current]
		self.l1 = float(tl1) / (tl1 + tl2 + tl3)
		self.l2 = float(tl2) / (tl1 + tl2 + tl3)
		self.l3 = float(tl3) / (tl1 + tl2 + tl3)
		for s1 in self.status | set(('BOS',)):
			for s2 in self.status | set(('BOS',)):
				for s3 in self.status:
					uni = self.l1 * self.tnt_div(self.uni[s3], self.tnt_total(self.uni))
					bi = self.l2 * self.tnt_div(self.bi[(s2, s3)], self.uni[s2])
					tri = self.l3 * self.tnt_div(self.tri[(s1, s2, s3)], self.bi[(s1, s2)])
					if uni + bi + tri == 0.0:
						self.trans[(s1, s2, s3)] = - 99999
					else:
						self.trans[(s1, s2, s3)] = log(uni + bi + tri)

		self.save(self.status, './Segmentation/checkpoints/tnt/status.pkl')
		self.save(self.wd, './Segmentation/checkpoints/tnt/wd.pkl')
		self.save(self.eos, './Segmentation/checkpoints/tnt/eos.pkl')
		self.save(self.eosd, './Segmentation/checkpoints/tnt/eosd.pkl')
		self.save(self.uni, './Segmentation/checkpoints/tnt/uni.pkl')
		self.save(self.bi, './Segmentation/checkpoints/tnt/bi.pkl')
		self.save(self.tri, './Segmentation/checkpoints/tnt/tri.pkl')
		self.save(self.word, './Segmentation/checkpoints/tnt/word.pkl')
		self.save(self.trans, './Segmentation/checkpoints/tnt/trans.pkl')

	def tnt_div(self, s1, s2):
		if s2 == 0:
			return 0
		return float(s1) / s2

	def tnt_total(self, d):
		return reduce(lambda x, y: x + y, map(lambda x: x[1], d.items()))

	def save(self, d, fname):
		pickle.dump(d, open(fname, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

	def geteos(self, tag):
		tmp = self.eosd.d[tag]
		if tmp == 0:
			return log(1.0 / len(self.status))
		if (tag, 'EOS') in self.eos.d:
			return log(self.eos.d[(tag, 'EOS')] / self.eosd.d[tag])
		else:
			return -99999

	def restore(self):
		self.status = pickle.load(open('./Segmentation/checkpoints/tnt/status.pkl', 'rb'))
		self.wd = pickle.load(open('./Segmentation/checkpoints/tnt/wd.pkl', 'rb'))
		self.eos = pickle.load(open('./Segmentation/checkpoints/tnt/eos.pkl', 'rb'))
		self.eosd = pickle.load(open('./Segmentation/checkpoints/tnt/eosd.pkl', 'rb'))
		self.uni = pickle.load(open('./Segmentation/checkpoints/tnt/uni.pkl', 'rb'))
		self.bi = pickle.load(open('./Segmentation/checkpoints/tnt/bi.pkl', 'rb'))
		self.tri = pickle.load(open('./Segmentation/checkpoints/tnt/tri.pkl', 'rb'))
		self.word = pickle.load(open('./Segmentation/checkpoints/tnt/word.pkl', 'rb'))
		self.trans = pickle.load(open('./Segmentation/checkpoints/tnt/trans.pkl', 'rb'))

	def cut(self, sentence):
		current = [(('BOS', 'BOS'), 0.0, [])]
		stage = {}
		for w in sentence:
			stage = {}
			samples = self.status
			if w in self.word:
				samples = self.word[w]
			for t in samples:
				wd = log(self.wd.d[(t, w)] / self.uni[t])
				for pre in current:
					p = pre[1] + wd + self.trans[(pre[0][0], pre[0][1], t)]
					if (pre[0][1], t) not in stage or p > stage[(pre[0][1], t)][0]:
						stage[(pre[0][1], t)] = (p, pre[2] + [t])
			stage = list(map(lambda x: (x[0], x[1][0], x[1][1]), stage.items()))
			current = heapq.nlargest(self.bean_search, stage, key=lambda x: x[1])
		current = heapq.nlargest(1, stage, key=lambda x: x[1] + self.geteos(x[0][1]))
		return zip(sentence, current[0][2])
