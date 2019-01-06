from collections import defaultdict


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
				current.append(tag)
				self.uni[(word, tag)] += 1
				self.bi[tuple(current[1:])] += 1
				self.tri[tuple(current)] += 1
				current.pop(0)

	def save(self, d, fname):
		pass

	def restore(self):
		pass

	def cut(self, sentence):
		pass

