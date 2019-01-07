from NLPs import NLPTools

text = "为了活跃课堂，你需要在网络上建一个“据点”，也就是说，需要建立一个网站，为学生提供课件，\
提醒任务完成时间，并且让学生之间建立联系。假设你是一位老师，并且已经有了自己的注册、评分和考勤系统，\
那就把这些运用到网络上吧。我们的目标是为学生建立一个平台，让他们能在现有的网络社交活动、\
与班上同学的交流和课堂作业之间自由转换。"

nlp = NLPTools(cut_fn='../Data/Segmentation/data.txt', cut_type='cbgm')

nlp.cutword('', True)
# seg_list = nlp.cutword(text)
# print(seg_list)
