from .Segmentation.seg import Seg


class NLPTools(object):
	def __init__(self, cut_fn, cut_type):
		# 断词
		self.cut_fn = cut_fn
		self.seg = Seg(cut_type)

	def cutword(self, doc, train=False):
		if not train:
			self.seg.restore()
			return [word for word in self.seg.cut(doc)]
		else:
			print("MSG : Start training segmentation service ...")
			self.seg.train(self.cut_fn)


