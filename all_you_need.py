import re
import math
import operator
import logging
import pickle
from tqdm import tqdm
import jieba
from jieba import analyse
import jieba.posseg as psg
import numpy as np
from string import punctuation as p
from collections import defaultdict

import nltk
from gensim import corpora, models
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cs

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

logging.basicConfig(level=logging.INFO)
p += '！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛' \
     '〜〝〞〟〰�〾〿–—‘・’‛“”„‟…‧﹏.□'


class Spec_Tokenizer(object):
    def __init__(self, num_words):
        self.num_words = num_words
        self.dict = defaultdict(int)
        self.token2idx = {}
        self.idx2token = {}

    def fit_on_texts(self, seg_list):
        for seg in seg_list:
            for word in seg:
                self.dict[word] += 1

        if self.num_words <= 0:
            self.dict = dict(sorted(self.dict.items(), key=operator.itemgetter(1), reverse=True))
        else:
            self.dict = dict(sorted(self.dict.items(), key=operator.itemgetter(1), reverse=True)[:self.num_words - 2])

        NULL = '--PAD--'
        OOV = '--OOV--'
        self.token2idx = {token: idx for idx, token in enumerate(self.dict, 2)}
        self.token2idx[NULL] = 0
        self.token2idx[OOV] = 1
        self.idx2token = {self.token2idx[token]: token for token in self.token2idx}
        logging.info('MSG : Tokenizer total {} words ...'.format(len(self.token2idx)))

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = []
            for word in text:
                if 'PAD' in word.upper():
                    word = '--PAD--'
                i = self.token2idx.get(word)
                if i is not None:
                    if i > len(self.token2idx):
                        raise ValueError('the word OutOfRange!')
                    else:
                        sequence.append(i)
                else:
                    sequence.append(self.token2idx['--OOV--'])
            sequences.append(sequence)

        return sequences

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = []
            for token in sequence:
                i = self.idx2token.get(token)
                if i is not None:
                    text.append(i)
                else:
                    text.append('--OOV--')
            texts.append(text)

        return texts

    @property
    def word_index(self):
        return self.token2idx


class nlp(object):
    """
    Pre-process for nlp tasks
    Include:
        - service
            - Preprocessing
                - build[cleaner + seg + shuffle + tag + dictionary + get tfidf + tokenize + padding] (clear)
                - seg (clear)
                - cleaner (clear)
                - get by pos-tag (clear)
                - tokenize[transform]（clear）
                - padding (clear)
                - shuffle data (clear)
                - get bow (clear)
                - get tf / idf (clear)
                - get ngram (clear)
                - new word discovery (clear)

            - Statistic
                - get cossim（clear）
                - get info entropy (clear)
                - get conditional entropy (clear)
                - get info gain (clear)
                - get chi (clear)
                - get gini coefficients (clear)
                - word count (clear)
                - get word mutual info (clear)

            - Tools
                - get stopwords (clear)
                - load embedding matrix (clear)

    """
    def __init__(self):
        logging.info('**Tool Guides**')
        fun_names = [name for name in dir(self) if '__' not in name]
        logging.info('This tool contains these features...')
        logging.info('function names:\n' + '\n'.join(fun_names))

    def build(self, corpus, num_words, seq_length, stopwords=None, specialwords=None, remove_alphas=False,
              remove_numbers=False, remove_urls=False, remove_punctuation=False, remove_email=False,
              remove_ip_address=False, keep_chinese_only=False, language='zh', shuffle=False, padding='post'):
        """
        采用通用语料来构建标准的数据处理框架，该语料作为基本的贝叶斯先验依据。
        :param corpus: 作为先验知识的语料库
        :param num_words: 最大字典数（负数和零表示不限长度）
        :param seq_length: 最大单句长度（供padding使用）
        :param stopwords: 停用词集列表
        :param specialwords: 特殊英文词集列表（为了减少字典复杂度，所有英文将被小写化，如果需保留特殊字符和构建词表）
        :param remove_alphas: 去除英文字符
        :param remove_numbers: 去除数字（包括电话，需要的特殊数字可加入特殊词中，会被转化为中文数字）
        :param remove_urls: 去除url网址
        :param remove_punctuation: 去除标点符号（大部分）
        :param remove_email: 去除email地址
        :param remove_ip_address: 去除ip地址
        :param keep_chinese_only: 只保留中文字符
        :param language: 待处理文本的语言（'zh' or 'en'）
        :param shuffle: 打乱先验文本顺序
        :param padding: 对先验本文长度进行补全和截断
        """
        if isinstance(corpus, str):
            if corpus == '':
                raise ValueError('Input must be at least one word!')
            self.corpus = [corpus]
        else:
            self.corpus = corpus

        self.remove_alphas = remove_alphas
        self.remove_urls = remove_urls
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.remove_email = remove_email
        self.remove_ip_address = remove_ip_address
        self.keep_chinese_only = keep_chinese_only

        self.num_words = num_words
        self.seq_length = seq_length
        self.language = language
        self.stopwords = stopwords
        self.specialwords = specialwords
        self.tfidf_vectorizer = TfidfVectorizer()

        logging.info('Clean and seg words ヾ(^Д^*)/')
        self.cleaned_corpus = [self.cleaner(sent,
                                            stopwords=self.stopwords,
                                            specialwords=self.specialwords,
                                            remove_alphas=remove_alphas,
                                            remove_numbers=remove_numbers,
                                            remove_urls=remove_urls,
                                            remove_punctuation=remove_punctuation,
                                            remove_email=remove_email,
                                            remove_ip_address=remove_ip_address,
                                            keep_chinese_only=keep_chinese_only) for sent in self.corpus]

        self.seg_list = [self.seg(sent, False, language) for sent in self.cleaned_corpus]
        if shuffle:
            self.seg_list = self.shuffle_data(self.seg_list)

        logging.info('Build vocabulary {dictionary} ...')
        self.dictionary = corpora.Dictionary(self.seg_list)

        logging.info('Calculate tf-idf features {tf/idf/tfidf_vectorizer} ...')
        self.tf = self.get_tf(self.seg_list)
        self.idf, self.default_idf = self.get_idf(self.seg_list)
        self.tfidf_vectorizer.fit([" ".join(seg) for seg in self.seg_list])

        logging.info('Build tokenizer {tokenizer/sequences/word_index} ...')
        self.tokenizer = Spec_Tokenizer(num_words=self.num_words)
        self.tokenizer.fit_on_texts(self.seg_list)
        self.sequences = self.transform(self.tokenizer, self.seg_list, reversed=False)
        self.word_index = self.tokenizer.word_index

        logging.info('Padding sequences {padded_sequences} ...')
        if padding is not None:
            self.padded_sequences = self.padding(self.sequences, self.seq_length, padding=padding)

    def seg(self, sentence, pos, language='zh'):
        """
        对文本（单句）进行分词操作。
        :param sentence: 待分词的文本
        :param pos: 是否带有tag
        :param language: 待处理文本的语言（'zh' or 'en'）
        :return: 文本的分词结果
        """
        if language == 'zh':
            if not pos:
                seg_list = jieba.cut(sentence, cut_all=False)
            else:
                seg_list = psg.cut(sentence)
            result = [item for item in seg_list if item != ' ']
        else:
            if not pos:
                result = sentence.split()
            else:
                result = nltk.pos_tag(sentence)
        return result

    def cleaner(self, text, stopwords=None, specialwords=None, remove_alphas=False, remove_numbers=False,
                remove_urls=False, remove_punctuation=False, remove_email=False, remove_ip_address=False,
                keep_chinese_only=False):
        """
        文本清洗工具。
        :param text: 单个待处理文本
        :param stopwords: 停用词集列表
        :param specialwords: 特殊英文词集列表（为了减少字典复杂度，所有英文将被小写化，如果需保留特殊字符和构建词表）
        :param remove_alphas: 去除英文字符
        :param remove_numbers: 去除数字（包括电话，需要的特殊数字可加入特殊词中，会被转化为中文数字）
        :param remove_urls: 去除url网址
        :param remove_punctuation: 去除标点符号（大部分）
        :param remove_email: 去除email地址
        :param remove_ip_address: 去除ip地址
        :param keep_chinese_only: 只保留中文字符
        :return: 清理完毕的单个文本
        """
        alphas = re.compile(r"[A-Za-z]+", re.IGNORECASE)
        numbers = re.compile(r"\d+", re.IGNORECASE)
        email = re.compile(r"[A-Za-z0-9\u4e00-\u9fa5]+@[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)+$", re.IGNORECASE)
        url_1 = re.compile(r"((https?|ftp|file):\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)", re.IGNORECASE)
        url_2 = re.compile(r"[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)", re.IGNORECASE)
        ips = re.compile(r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}(:)?[0-9]{1,8}", re.IGNORECASE)

        chinese_pattern = re.compile(u'[\u4e00-\u9fa5]+')
        stop_p = p + "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰�〾〿–—‘・’‛“”„‟…‧﹏."
        num_maps = {'{}'.format(i): '{}'.format(j) for i, j in zip(range(10), ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖'])}

        # 关键成分过滤
        text = re.sub(r'\u0020+', ' ', text)
        text = re.sub(r'\xa0+', ' ', text)

        if keep_chinese_only:
            text = " ".join(chinese_pattern.findall(text))
        else:
            # 英文规范
            text = re.sub(r"what's", "what is", text)
            text = re.sub(r"\'s", " is", text)
            text = re.sub(r"\'ve", " have", text)
            text = re.sub(r"can't", "cannot", text)
            text = re.sub(r"n't", " not", text)
            text = re.sub(r"i'm", "i am", text)
            text = re.sub(r"i’m", "i am", text)
            text = re.sub(r"\'re", " are", text)
            text = re.sub(r"\'d", " would", text)
            text = re.sub(r"\'ll", " will", text)

            text = text.lower()
            if specialwords is not None:
                for word in specialwords:
                    if word in text:
                        if numbers.match(word):
                            new_word = ""
                            for char in word:
                                tmp = num_maps.get(char)
                                if tmp is not None:
                                    new_word += tmp
                                else:
                                    new_word += char
                            text = text.replace(word, new_word)
                        else:
                            text = text.replace(word, word.capitalize())

        if remove_urls:
            text = url_1.sub(' ', text)
            text = url_2.sub(' ', text)
        if remove_ip_address:
            text = ips.sub(' ', text)
        if remove_email:
            text = email.sub(' ', text)
        if remove_alphas:
            text = alphas.sub(' ', text)
        if remove_numbers:
            text = numbers.sub(' ', text)

        # 去标点
        if remove_punctuation:
            for punc in stop_p:
                if punc in text:
                    text = text.replace(punc, ' ')

        # 去停用词
        if stopwords is not None:
            text = "".join([char for char in text if char not in stopwords])

        text = re.sub(r" +", " ", text)

        return text

    def get_by_postag(self, sentence, tags, language):
        """
        根据所需的tag获取句子的词。
        :param sentence: 待处理的句子（单个）
        :param tags: 需要获取的tag种类集合
        :param language: 待处理文本的语言（'zh' or 'en'）
        :return: 过滤指定tag保留的句子结构
        """
        seg_list = self.seg(sentence, pos=True, language=language)
        filter_list = []

        if language == 'zh':
            for seg in seg_list:
                word = seg.word
                flag = seg.flag

                for tag in tags:
                    if flag.startswith(tag):
                        filter_list.append(word)
                        break
        else:
            for word, flag in seg_list:
                for tag in tags:
                    if flag.startswith(tag):
                        filter_list.append(word)
                        break

        return filter_list

    def get_cossim(self, sent1, sent2):
        """
        计算两个句子之间的余弦相似度。
        :param sent1: 句子1
        :param sent2: 句子2
        :return: 两个句子的余弦相似度
        """
        if isinstance(sent1, str):
            sent1 = self.cleaner(sent1,
                                 stopwords=self.stopwords,
                                 specialwords=self.specialwords,
                                 remove_alphas=self.remove_alphas,
                                 remove_numbers=self.remove_numbers,
                                 remove_urls=self.remove_urls,
                                 remove_punctuation=self.remove_punctuation,
                                 remove_email=self.remove_email,
                                 remove_ip_address=self.remove_ip_address,
                                 keep_chinese_only=self.keep_chinese_only)

            seg_sent1 = [" ".join(self.seg(sent1, pos=False))]
        else:
            raise ValueError('Please input a str format sentence (´▽`)ﾉ ')
        if isinstance(sent2, str):
            sent2 = self.cleaner(sent2,
                                 stopwords=self.stopwords,
                                 specialwords=self.specialwords,
                                 remove_alphas=self.remove_alphas,
                                 remove_numbers=self.remove_numbers,
                                 remove_urls=self.remove_urls,
                                 remove_punctuation=self.remove_punctuation,
                                 remove_email=self.remove_email,
                                 remove_ip_address=self.remove_ip_address,
                                 keep_chinese_only=self.keep_chinese_only)

            seg_sent2 = [" ".join(self.seg(sent2, pos=False))]
        else:
            raise ValueError('Please input a str format sentence (´▽`)ﾉ ')
        if self.tfidf_vectorizer is None:
            raise ValueError("Please build tfidf_vectorizer with corpus...")
        s1_matrix = self.tfidf_vectorizer.transform(seg_sent1)
        s2_matrix = self.tfidf_vectorizer.transform(seg_sent2)
        return cs(s1_matrix, s2_matrix).flatten()[0]

    def get_info_entropy(self, array):
        """
        计算文本词（或特征）对类别分布结果影响的信息熵值。
        :param array: 所有样本的类别
        :return: 信息熵值
        """
        if isinstance(array, list):
            array = np.array(array)

        x_value_list = np.unique(array)
        ent = 0.0
        for x_value in x_value_list:
            p = float(array[array == x_value].shape[0] / array.shape[0])
            logp = np.log2(p)
            ent -= p * logp

        return ent

    def get_cond_entropy(self, array_x, array_y):
        """
        计算文本词（或特征）对类别分布结果影响的条件熵值。
        :param array_x: 所有样本是否含有该词（或该特征）
        :param array_y: 所有样本的类别
        :return: 条件熵值
        """
        if isinstance(array_x, list):
            array_x = np.array(array_x)
        if isinstance(array_y, list):
            array_y = np.array(array_y)

        x_value_list = np.unique(array_x)
        ent = 0.0
        for x_value in x_value_list:
            sub_y = array_y[array_x == x_value]
            tmp_ent = self.get_info_entropy(sub_y)
            ent -= (float(sub_y.shape[0]) / array_y.shape[0]) * tmp_ent

        return ent

    def get_info_gain(self, array_x, array_y):
        """
        计算文本词（或特征）对类别分布结果影响的信息增益。
        :param array_x: 所有样本是否含有该词（或该特征）
        :param array_y: 所有样本的类别
        :return: 信息增益值
        """
        if isinstance(array_x, list):
            array_x = np.array(array_x)
        if isinstance(array_y, list):
            array_y = np.array(array_y)

        base_ent = self.get_info_entropy(array_y)
        cond_ent = self.get_cond_entropy(array_x, array_y)
        ent_grap = base_ent - cond_ent

        return ent_grap

    def get_chi(self, array_x, array_y):
        """
        计算文本词对于分布的卡方检定系数。（越大表示对于类别分布结果越相关）
        :param array_x: 各文档是否含有该词（0和1）
        :param array_y: 各文档的类别
        :return: 卡方检定系数
        """
        if isinstance(array_x, list):
            array_x = np.array(array_x)
        if isinstance(array_y, list):
            array_y = np.array(array_y)

        y_value_list = np.unique(array_y)
        x_value_list = np.unique(array_x)
        if len(x_value_list) != 2 or (0 not in x_value_list or 1 not in x_value_list):
            raise ValueError('array_x must composed with 0 and 1 means whether it contains the word or not (Ｔ▽Ｔ)')

        cls_chi = []
        for cls in y_value_list:
            cls_total_count = array_y[array_y == cls].shape[0]
            sub_y = array_y[array_x == 1]
            doc_count = sub_y.shape[0]

            a = sub_y[sub_y == cls].shape[0]
            b = doc_count - a
            c = cls_total_count - a
            d = array_y.shape[0] - doc_count - cls_total_count + a

            chi = array_y.shape[0] * (a * d - b * c) ** 2 / ((a + c) * (a + b) * (b + d) * (c + d))
            cls_chi.append(chi)

        return cls_chi

    def get_gini_coefficients(self, array):
        """
        计算序列的基尼系数。（越接近0表示分布越均匀）
        :param array: 带计算的序列
        :return: 序列分布的基尼系数
        """
        mad = np.abs(np.subtract.outer(array, array)).mean()
        rmad = mad / np.mean(array)
        g = 0.5 * rmad
        return g

    def word_count(self, corpus, word_dict=None):
        """
        统计数据集中的总词频（相当于reduce tf）。
        :param corpus: 待统计的数据集
        :param word_dict: 是否将词频添加到已有的词典（扩张）
        :return: 统计后的词典（已排序）
        """
        if isinstance(corpus, str):
            raise ValueError('Please make corpus segmented ￣ω￣=')
        elif len(np.array(corpus).shape) == 1:
            corpus = [corpus]

        array_shape = np.array(corpus).shape
        word_dict = {} if word_dict is None else word_dict
        if len(array_shape) == 2:
            for sent in corpus:
                for word in sent:
                    if word not in word_dict:
                        word_dict[word] = 0
                    word_dict[word] += 1
        elif len(array_shape) == 1:
            for word in corpus:
                if word not in word_dict:
                    word_dict[word] = 0
                word_dict[word] += 1
        word_dict = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)

        return dict(word_dict)

    def get_tf(self, corpus):
        """
        计算文本数据集的文档词频。
        :param corpus: 文本数据集
        :return: 文本数据集统计后的tf值
        """
        if isinstance(corpus[0], str):
            corpus = [corpus]
        tf_list = []

        for doc in corpus:
            tf_dic = defaultdict(int)
            tt_count = len(doc)
            for word in doc:
                tf_dic[word] += 1

            for k, v in tf_dic.items():
                tf_dic[k] = float(v) / tt_count

            tf_list.append(tf_dic)

        return tf_list

    def get_idf(self, corpus):
        """
        计算文本数据集词出现的逆文档频率。
        :param corpus: 文本数据集
        :return: 文本数据集统计后的idf值
        """
        if isinstance(corpus[0], str):
            corpus = [corpus]
        df_dic, idf_dic = defaultdict(int), defaultdict(int)
        tt_count = len(corpus)

        for doc in corpus:
            for word in set(doc):
                df_dic[word] += 1

        for k, v in df_dic.items():
            idf_dic[k] = math.log(tt_count) - math.log(v + 1)

        default_idf = math.log(tt_count / 1.0)

        return idf_dic, default_idf

    def get_bow(self, corpus):
        """
        获取句子的词袋表示（BOW）,结果为gensim格式的bow表征（词id，数量）的tuple。
        :param corpus: 待处理的句子
        :return: 句子集合的词袋表示
        """
        if self.dictionary is None:
            raise ValueError("Please build dictionary for corpus ...")

        return self.dictionary.doc2bow(corpus)

    def get_ngram(self, text, n):
        """
        获取数据（单句）的n-gram表示。
        :param text: 待处理的文本数据
        :param n: n-gram的n值
        :return: 文本数据集的n-gram表示
        """
        if not isinstance(text, str):
            raise ValueError('Something wrong when load corpus during get ngram (╯﹏╰)b')

        split_text = list(text)
        loop = len(split_text) + 1 - n
        result = []
        for i in range(loop):
            result.append(text[i: i+n])

        return result

    def get_mutual_info(self, dictionary, word_a, word_b):
        """
        计算两个词之间的互信息。
        :param dictionary: 统计生成的先验词典（计数字典）
        :param word_a: 词1
        :param word_b: 词2
        :return: 两个词之间的互信息值
        """
        total_words = len(dictionary)
        whole_word = word_a + word_b

        freq_a = dictionary.get(word_a, 0) / total_words
        freq_b = dictionary.get(word_b, 0) / total_words
        freq_whole = dictionary.get(whole_word, 0) / total_words
        if freq_a * freq_b == 0:
            raise ValueError('There are no words in the dictionary ╮(╯﹏╰）╭')

        return freq_whole * np.log2(freq_whole / (freq_a * freq_b)) if not np.isnan(freq_whole * np.log2(freq_whole / (freq_a * freq_b))) else 0.

    def new_words_discovery(self, corpus, max_gram=4, min_entropy=1.5, min_p=7, min_count=20, min_reback=0.3):
        """
        利用互信息和信息熵挖掘新词组合（BPE尝试）。
        :param corpus: 待挖掘的文本数据集
        :param max_gram: 最大可挖掘的词组合长度
        :param min_entropy: 最小满足的信息熵
        :param min_p: 最小满足的互信息系数
        :param min_count: 最小出现词频
        :param min_reback: 最小热门词词频比率
        :return: 新词列表
        """
        if isinstance(corpus, list):
            corpus = "".join(corpus)
        elif isinstance(corpus, np.ndarray) and len(corpus.shape) == 2:
            corpus = list(np.concatenate(corpus, axis=0))
        elif isinstance(corpus, str):
            corpus = corpus
        else:
            raise ValueError('Something wrong when load corpus during get ngram (╯﹏╰)b')

        word_freqs = {0: {}}
        for i in range(1, 3):
            tmp_list = self.get_ngram(corpus, i)
            tmp_freqs = self.word_count(tmp_list)
            word_freqs[i] = tmp_freqs

        def __get_pro(word, dictionary):
            len_word = len(word)
            total_words = sum(dictionary[len_word].values())
            pro = dictionary[len_word][word] / total_words
            return pro

        candidate_words = []
        for word, count in word_freqs[2].items():
            if count < min_count:
                continue
            tmp_p = min([__get_pro(word[:t], word_freqs) * __get_pro(word[t:], word_freqs) for t in range(1, len(word))])
            if np.log2(__get_pro(word, word_freqs) / tmp_p) > min_p:
                candidate_words.append(word)

        candidate_words = list(set(candidate_words))
        results = []

        reback_words = []

        def __loop(word, direct='both'):
            try:
                lr = re.findall(r'(.)?%s(.)?' % word, corpus)
            except:
                return
            left_list = [w[0] for w in lr if w[0] != '']
            right_list = [w[1] for w in lr if w[1] != '']
            left_entropy = self.get_info_entropy(np.array(left_list))
            right_entropy = self.get_info_entropy(np.array(right_list))

            if min(left_entropy, right_entropy) > min_entropy:
                results.append(word)

            if direct != 'right'and left_entropy <= min_entropy and len(word) < max_gram:
                for w, count in self.word_count(left_list).items():
                    if len(left_list) >= min_count / 2 and count / len(left_list) > min_reback:
                        reback_words.append((w + word, 'left'))
                    else:
                        break
            if direct != 'left'and right_entropy <= min_entropy and len(word) < max_gram:
                for w, count in self.word_count(right_list).items():
                    if len(right_list) >= min_count / 2 and count / len(right_list) > min_reback:
                        reback_words.append((word + w, 'right'))
                    else:
                        break

        for word in candidate_words:
            __loop(word)
            while len(reback_words) > 0:
                word, direction = reback_words.pop(0)
                __loop(word, direction)

        return results

    def shuffle_data(self, corpus):
        """
        对数据集顺序进行打乱
        :param corpus: 待处理的数据集
        :return: 打乱顺序后的数据集
        """
        if isinstance(corpus, list):
            corpus = np.array(corpus)
        indices = np.arange(len(corpus))
        shuffled_indices = np.random.permutation(indices)
        return corpus[shuffled_indices]

    def transform(self, tokenizer, corpus, reversed=False):
        """
        文本tokenizer（对文本进行标签化）
        :param tokenizer: 标签化工具物件（工具自带的分装类）
        :param corpus: 待处理的文本数据集（已经分词）
        :param reversed: True表示从标签转换成文字，False表示从文本转换成标签数字
        :return: 转换后的文本数据
        """
        if reversed:
            return tokenizer.sequences_to_texts(corpus)
        else:
            return tokenizer.texts_to_sequences(corpus)

    def padding(self, corpus, seq_length, padding='post'):
        """
        为数据集进行长度的规范。
        :param corpus: 待规范的数据集
        :param seq_length: 标准长度
        :param padding: 填充和截断的位置（post为右边处理，pre为左边）
        :return:
        """
        if isinstance(corpus, str):
            corpus = [corpus]
        return pad_sequences(corpus, maxlen=seq_length, padding=padding, truncating=padding)

    def get_stopwords(self, filename, encoding='utf-8'):
        """
        加载本地停用词表。
        :param filename: 预读取的停用词文件名
        :param encoding: 文件编码格式
        :return: 停用词集的list
        """
        return [re.sub(r' +', '', sw.replace('\ufeff', '')) for sw in open(filename, 'r', encoding=encoding).read().splitlines()]

    def load_embedding_matrix(self, dictionaries, embedding_size, embedding_file_path, dtype='other'):
        """
        加载预训练的词嵌入（embedding）矩阵。
        :param dictionary: 待加载词向量的词集
        :param embedding_size: 词嵌入维度大小
        :param embedding_file_path: 预训练的词向量文档
        :param type: 词嵌入加载方式（'self'：通过gensim word2vec训练的加载方式，'other'：标准的词向量文档（非二进制文档））
        :return: 词向量矩阵
        """
        def __get_matrix(dict, embed_dict):
            nb_words = len(dict)
            embedding_matrix = np.zeros((nb_words + 2, embedding_size))
            for idx, word in tqdm(dict.items()):
                if word in embed_dict:
                    embedding_vector = embed_dict.get(word)
                    embedding_matrix[idx] = embedding_vector
                else:
                    embedding_vector = np.random.uniform(low=-0.2, high=0.2, size=embedding_size)
                    embedding_matrix[idx] = embedding_vector
            return embedding_matrix

        print('Creating embedding matrix ...')
        if embedding_file_path is not None:
            assert embedding_size is not None
            embedding_dict = {}
            if dtype == 'self':
                model = word2vec.Word2Vec.load(embedding_file_path)
                for word in model.wv.vocab.keys():
                    embedding_dict[word] = model.wv[word]
            else:
                with open(embedding_file_path, 'r', encoding='utf-8') as f:
                    for line in tqdm(f):
                        try:
                            array = line.strip().split()
                            word = "".join(array[:-embedding_size])
                            vector = list(map(float, array[-embedding_size:]))
                            embedding_dict[word] = vector
                        except:
                            continue

            if len(dictionaries) > 1:
                embedding_matrices = []
                for dictionary in dictionaries:
                    matrix = __get_matrix(dictionary, embedding_dict)
                    embedding_matrices.append(matrix)
                return embedding_matrices
            else:
                return __get_matrix(dictionaries, embedding_dict)

        else:
            if len(dictionaries) > 1:
                embedding_matrices = []
                for dictionary in dictionaries:
                    nb_words = len(dictionary)
                    embedding_matrices.append(np.random.uniform(low=-0.2, high=0.2, size=(nb_words + 2, embedding_size)))
                return embedding_matrices
            else:
                nb_words = len(dictionaries)
                return np.random.uniform(low=-0.2, high=0.2, size=(nb_words + 2, embedding_size))


class BM25_ranker(object):
    def __init__(self, corpus, PARAM_K1=1.5, PARAM_B=0.75, EPSILON=0.25, train=True, param_path=None):
        self.k1 = PARAM_K1
        self.b = PARAM_B
        self.epsilon = EPSILON

        self.corpus_size = len(corpus)
        self.avgdl = sum(map(lambda x: float(len(x)), corpus)) / self.corpus_size
        self.corpus = corpus
        self.path = param_path

        if train:
            self.f = []
            self.df = defaultdict(int)
            self.idf = {}
            self.initialize()
        elif param_path is not None:
            load_params = pickle.load(open(param_path, 'rb'))
            self.f = load_params['f']
            self.df = load_params['df']
            self.idf = load_params['idf']
        else:
            raise ValueError('Please set the pretrained model or select train equals True...')

    def initialize(self):
        for document in self.corpus:
            frequencies = defaultdict(int)
            for word in document:
                frequencies[word] += 1
            self.f.append(frequencies)

            for word in frequencies.keys():
                self.df[word] += 1

        for word, freq in self.df.items():
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

        self.save_params(self.path)

    def save_params(self, path):
        param_dict = {'f': self.f, 'df': self.df, 'idf': self.idf}
        pickle.dump(param_dict, open(path, 'wb'))

    def get_score(self, document, index, average_idf):
        score = 0
        for word in document:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else self.epsilon * average_idf
            score += (idf * self.f[index][word] * (self.k1 + 1) /
                      (self.f[index][word] + self.k1 * (1 - self.b + self.b * self.corpus_size / self.avgdl)))

        return score

    def get_topk(self, document, average_idf, k):
        scores = {}
        for index, corpus in enumerate(self.corpus):
            score = self.get_score(document, index, average_idf)
            scores[index] = score

        results = [i for i in sorted(scores.items(), key=operator.itemgetter(1), reverse=True)[:k]]
        return results


class algorithms(object):
    """
    Traditional machine/deep learning algorithms for nlp
    Include:
        - keyword extraction
        - document matching
    """
    def __init__(self, nlp_tools):
        self.nlp_tools = nlp_tools

        logging.info('**Tool Guides**')
        fun_names = [name for name in dir(self) if '__' not in name and name != 'nlp_tools']
        logging.info('This tool contains these features...')
        logging.info('function names:\n' + '\n'.join(fun_names))

    def build_func(self, task, num_topic=3):
        if len(self.nlp_tools.seg_list) == 0:
            raise ValueError('Please build nlp_tools for a based corpus and try again (￣▽￣)~*')

        if task == 'keyword extraction':
            self.idf_dic, self.default_idf = self.nlp_tools.idf, self.nlp_tools.default_idf
            corpus = [self.nlp_tools.get_bow(seg) for seg in self.nlp_tools.seg_list]

            self.tfidf_model = models.TfidfModel(corpus)
            self.corpus_tfidf = self.tfidf_model[corpus]
            self.dictionary = self.nlp_tools.dictionary

            self.lsi_model = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=num_topic)
            self.lda_model = models.LdaModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=num_topic)

            self.lsi_wordtopic_dic = self.__get_word_topic_dict(self.dictionary, self.tfidf_model, self.lsi_model)
            self.lda_wordtopic_dic = self.__get_word_topic_dict(self.dictionary, self.tfidf_model, self.lda_model)
        else:
            pass

    # 关键词提取(一句话粒度)
    def keyword_extraction(self, seg_list, keyword_num=2, algorithm='tfidf', stopword_file=None):
        if algorithm == 'tfidf':
            return self.__tfidf_keyword_extractor(seg_list, keyword_num)

        elif algorithm == 'textrank':
            return self.__textrank_keyword_extractor(seg_list, keyword_num, stopword_file)

        else:
            return self.__topicmodel_keyword_extractor(seg_list, keyword_num, model_type=algorithm)

    def __tfidf_keyword_extractor(self, seg_list, keyword_num):
        tfidf_dic = {}
        tf_dic = self.nlp_tools.get_tf(seg_list)[0]

        for word in seg_list:
            idf = self.idf_dic.get(word, self.default_idf)
            tf = tf_dic.get(word)
            tfidf = tf * idf
            tfidf_dic[word] = tfidf

        return [k for k, _ in sorted(tfidf_dic.items(), key=operator.itemgetter(1), reverse=True)][:keyword_num]

    def __textrank_keyword_extractor(self, seg_list, keyword_num, stopword_file=None):
        if stopword_file is not None:
            analyse.set_stop_words(stopword_file)
        textrank = analyse.textrank

        return textrank("".join(seg_list), keyword_num)

    def __topicmodel_keyword_extractor(self, seg_list, keyword_num, model_type='lda'):
        if model_type == 'lsi':
            model = self.lsi_model
            wordtopic_dic = self.lsi_wordtopic_dic
        elif model_type == 'lda':
            model = self.lda_model
            wordtopic_dic = self.lda_wordtopic_dic
        else:
            raise ValueError('There is no model type you need, please select from tfidf, textrank, lda or lsi o(TωT)o')

        return self.__get_topic_sim(seg_list, self.dictionary, self.tfidf_model, model, wordtopic_dic)[:keyword_num]

    def __get_word_topic_dict(self, dictionary, tfidf_model, model):
        wordtopic_dic = {}
        word_dict = dictionary.token2id

        for word, _ in word_dict.items():
            single_word = [word]
            wordcorpus = tfidf_model[dictionary.doc2bow(single_word)]
            wordtopic = model[wordcorpus]
            wordtopic_dic[word] = wordtopic

        return wordtopic_dic

    def __get_topic_sim(self, seg_list, dictionary, tfidf_model, model, wordtopic_dic):
        sentcorpus = tfidf_model[dictionary.doc2bow(seg_list)]
        senttopic = model[sentcorpus]

        sim_dict = {}
        for word in seg_list:
            if word in wordtopic_dic:
                word_topic = wordtopic_dic[word]
                sim = cs([[item[1] for item in word_topic]], [[item[1] for item in senttopic]])
                sim_dict[word] = sim[0][0]

        return [k for k, _ in sorted(sim_dict.items(), key=operator.itemgetter(1), reverse=True)]

    def document_matching(self, seg_list, topk, reuse=False, param_path=None):
        if not reuse and param_path is None:
            raise ValueError('Please input the param_path if set reuse == False (￣▽￣)~*')

        if len(self.nlp_tools.seg_list) == 0:
            raise ValueError('Please build nlp_tools for a based corpus and try again (￣▽￣)~*')

        ranker = BM25_ranker(self.nlp_tools.seg_list, train=not reuse, param_path=param_path)
        indices = ranker.get_topk(seg_list, self.nlp_tools.default_idf, topk)
        sents, scores = [tup[0] for tup in indices], [tup[1] for tup in indices]

        return sents, scores

