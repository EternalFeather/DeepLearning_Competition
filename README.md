Natural Language Processing(NLP)
===
# 知识总览
- 句法语义分析
    - 分词（SEGMENTATION）
    - 词性标注（POS_TAG）
    - 命名实体识别（NER）
    - 句法分析（SEMENTIC PARSER）
    - 语义角色识别
    - 多义词消岐
    - 指代消解
    - 句法纠错
- 关键词抽取
    - 实体识别
    - 时间抽取
    - 因果关系抽取
- 文本挖掘
    - 文本聚类
    - 文本分类
    - 信息抽取
    - 文本摘要
    - 情感分析
    - 知识图谱
- 机器翻译
- 信息检索
    - 文本匹配
    - 文本排序
- 问答系统
    - [信息检索]
    - 语义分析
        - 实体链接
        - 关系识别
        - 知识库检索
        - 文本排序
- 对话系统
    - 意图理解
    - [问答系统]
    - 聊天引擎
    - 对话管理
    - 对话生成

# 常用中文语料
- 中文维基百科
    - [Linke](https://dumps.wikimedia.org/zhwiki/)
- 搜狗新闻
    - [Link](http://download.labs.sogou.com/resource/ca.php)
- IMDB情感分析语料库
    - [Link](https://www.kaggle.com/tmdb/tmdb-movie-metadata)
- 句法分析数据集
    - 中文宾州树库（CTB）
    - 清华树库（TCT）
    - 台湾中研院树库（STB）

# 知识细化
## 分词（W_SEG)

### 规则分词

- 正向最大匹配法
- 逆向最大匹配法
- 双向最大匹配法

### 统计分词

- 语言模型
![66abf31a-25fd-4b18-8496-093a126c7966.png](README_files/66abf31a-25fd-4b18-8496-093a126c7966.png)
常用n-gram模型来避免长句子带来的维度灾难；同时利用MLE似然评估方法来减少n-gram的参数数量，把统计问题上升到了训练预测问题。
- HMM（隐形马尔科夫）
将分词当做序列标注问题，借助控制转移概率的方法来排除贝叶斯算法带来的顺序不合理（标注问题中体现）情况。常用齐次马尔科夫简化计算。
    - 优化算法：Veterbi算法（计算每个时间点的时候只取发射后验概率最大的那一个进行传递--区别于前向后向算法）
- CRF（条件随机场）
改进马可夫假设中当前状态只与前一个时间点有关，转而变成全局状态。
- 神经网络
- 混合分词算法

### 常用分词工具

- Jieba
- pkuseg（北大分词）

## 命名实体识别（NER）

### 规则识别

- 基于实体词典和命名规则

## 统计命名实体识别

- HMM
- 最大熵（EM）
- 条件随机场（CRF）
    - 针对HMM在多重交互特征场景下的限制而提出的方法
    - 将状态矩阵和转移矩阵都实例化成为函数，在优化的过程中考虑全局特征信息

## 关键词提取（KW-extraction）

### 有监督

- 文本分类

## 无监督

- TF_IDF
    - IDF计算过程中采用拉普拉斯平滑，避免分母出现零的情况
- TextRank
    - 脱离文本依赖，仅需要单篇文本处理
![5e36aa87-869a-45d9-b78a-bddb41e156b2.png](README_files/5e36aa87-869a-45d9-b78a-bddb41e156b2.png)
    - 
![aef16f1f-8136-4c13-b9b6-c16e6c7a3b70.png](README_files/aef16f1f-8136-4c13-b9b6-c16e6c7a3b70.png)

- 主题模型
    - 挖掘隐含的关键词
    - LSA【LSI】（频度学派，主要采用SVD暴力破解）
    - pLSA（利用EM算法优化代替SVD暴力破解）
    - LDA（贝叶斯学派拟合分布）
        - 优化采用Gibbs sampling

### 常用算法工具

- Jieba analyse 库的textrank
- Gensim models 库的lsamodel、ldamodel

## 句法分析（Syntactic analysis）

- 数据结构 ： 句法树
- 难点
    - 歧义
    - 搜索空间大

### 基于规则的句法分析

- 人工整理的大量句法模板（实用性弱）

### 基于统计的句法分析

- 实质任务是给候选句法树打分
- 主要算法包括：
    - Probabilistic Context Free Grammar（PCFG）
        - 基于上下文无关的概率短语结构分析器
        - 是一种生成式方法
        - 基本结构无元组：（X；V；S；R；P）
        - 基本步骤1、计算句子的句法树概率P：采用内向（bottom_up、同一级节点之间）、外向算法（top_down、不同级节点之间）
        - 基本步骤2、选择最佳的句法树：采用Viterbi算法
        - 基本步骤3、优化使得最优句法树的概率最大：采用EM算法
    - 基于最大间隔马尔可夫网络（Max_Margin Markov Networks）
        - 用以解决**结构化预测**问题
        - 通常采用多个二元分类（分类短语标记）来取代多元分类
    - 基于CRF的句法分析
        - 将上述的马可夫计算改成条件随机场，标注短语序列
        - 与PCFG不同之处在于：1、计算的是条件概率而非联合概率 2、需要对概率进行归一化
    - 基于移进——归约的句法分析（Shift_Reduce Algorithm）
        - 采用堆栈的数据结构进行迭代

### 常用工具

- Stanford Parser 的PCFG句法分析器 （python接口需要nltk和java jdk）

## 文本表征（Text tokenization / vectorization）

- 文本表征除了词统计表征，还可以采用向量化表示（字符、词、句子、段落层面都有）

### 文本向量化算法

- Bag Of Word
    - 通过统计次数以字典的形式表示
    - 缺点：
        - 维度灾难（Dimensional Catastrophe）
        - 词序信息遗漏
        - 语义鸿沟
- Word2vec
    - 理论支持：Distributional hypothesis（分布假说），即上下文类似的词本身语义也相似
    - 属于VSM（vector space machine）范畴

## 知识图谱（KG）

### Knowledge Representation Learning

- TransE
- TransR
- TransH
- TransG
- KG2E
- ManifoldE

### Neural Relation Extraction

- Sentence Level NER
    - Input Encoder
        - word embedding
        - position embedding
        - part-of-speech tag embedding
        - wordnet hypernym embedding
    - Sentence Encoder
        - Convolution neural network encoder（CNN + maxpool）
        - Recurrent nerual network encoder（RNN + maxpool）
        - Dependency tree-structured LSTM（【Child-Sum / N-ary】 Tree-LSTM）
        - Recrusive neural network encoder（MV-RNN）
    - Relation Classifier
        - （Softmax + Dense）

- Document Level NER
    - Input Encoder
    - Sentence Encoder
    - Document Encoder
        - Random Encoder
        - Max Encoder
        - Average Encoder
        - Attentive Encoder
    - Relation Classifier

### Bridging Knowledge with Text: Entity Linking

- The Entity Linking Framework
    - Name Mention Identificaion
        - Name Entity Recongnition（NER）
        - Dictionary-based matching

# Reference

Deep learning natural language processing

# KeyWords

###### tags: `NLP`