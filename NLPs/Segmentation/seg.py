from NLPs.Segmentation.tnt import Tnt
from NLPs.Segmentation.cbgm import Cbgm


class Seg(object):
	def __init__(self, name='cbgm'):
		if name == 'tnt':
			self.segger = Tnt()
		else:
			self.segger = Cbgm()

	def train(self, fname):
		f = open(fname, 'r', encoding='utf-8')
		data = []
		for line in f:
			line = line.strip()
			if not line:
				continue
			tmp = map(lambda x: x.split('/'), line.split(' '))
			data.append(tmp)
		f.close()
		self.segger.train(data)
		print('Done!')

	def restore(self):
		self.segger.restore()

	def cut(self, sentence):
		ret = self.segger.cut(sentence)
		tmp = ""
		for i in ret:
			if i[1] == 'e':
				yield tmp + i[0]
				tmp = ""
			elif i[1] == 'b' or i[1] == 's':
				if tmp:
					yield tmp
				tmp = i[0]
			else:
				tmp += i[0]
		if tmp:
			yield tmp
