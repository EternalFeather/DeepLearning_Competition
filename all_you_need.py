# package area
import os
import re
import jieba
import nltk
import math
import operator
import logging
from jieba import analyse
import jieba.posseg as psg
from string import punctuation as p
from collections import defaultdict

from gensim import corpora, models
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cs

from tensorflow.keras.preprocessing.sequence import pad_sequences

logging.basicConfig(level=logging.INFO)
p += '！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰�〾〿–—‘・’‛“”„‟…‧﹏.'


class Spec_Tokenizer(object):
    def __init__(self, num_words):
        self.num_words = num_words
        self.dictionary = defaultdict(int)
        self.token2idx = {}
        self.idx2token = {}

    def fit_on_texts(self, seg_list):
        for seg in seg_list:
            for word in seg:
                self.dictionary[word] += 1

        self.dictionary = sorted(self.dictionary.items(), key=operator.itemgetter(1), reversed=True)[:self.num_words - 2]

        NULL = '--PAD--'
        OOV = '--OOV--'
        self.token2idx = {token[0]: idx for idx, token in enumerate(self.dictionary, 2)}
        self.token2idx[NULL] = 0
        self.token2idx[OOV] = 1
        self.idx2token = {token2idx[token]: token for token in token2idx}

    def texts_to_sequences(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        sequences = []
        for text in texts:
            sequence = []
            for word in text:
                if 'PAD' in word.upper():
                    word = '--PAD--'
                i = self.token2idx.get(word)
                if i is not None:
                    if i >= self.num_words:
                        raise ValueError('the word OutOfRange!')
                    else:
                        sequence.append(i)
                else:
                    sequence.append(self.token2idx['--OOV--'])
            sequences.append(sequence)

        return sequences

    def sequences_to_texts(self, sequences):
        if isinstance(texts, str):
            texts = [texts]

        texts = []
        for sequence in sequences:
            text = []
            for token in sequence:
                i = self.idx2token.get(token)
                if i is not None:
                    if i >= self.num_words:
                        raise ValueError('the word OutOfRange!')
                    else:
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
        - single sentence (pairs) service
            - seg
            - cleaner
            - get by pos-tag
            - get cossim（based on corpus）
            - get info entropy
            - get gini coefficients

        - all corpus service
            - word count
            - get tf / idf
            - get bow
            - shuffle data
            - tokenize（text_to_sequence / sequence_to_text）
            - padding
            - make batch

        - data based
            - get stopwords
            - load embedding matrix

    """
    def __init__(self):
        logging.info('**Tool Guides**')
        fun_names = [name for name in dir(self) if '__' not in name]
        logging.info('This tool contains these features...')
        logging.info('function names:\n' + '\n'.join(fun_names))

    def build(self, corpus, embedding_size, num_words, seq_length, embedding_file_path=None, emb_type='self',
              stopwords=None, specialwords=None, remove_alphas=False, remove_numbers=False, remove_urls=False,
              remove_punctuation=False, remove_ip_address=False, language='zh', shuffle=False, padding=True):
        if isinstance(corpus, str):
            if corpus == '':
                raise ValueError('Input must be at least one word!')
            self.corpus = [corpus]
        else:
            self.corpus = corpus

        self.embedding_size = embedding_size
        self.num_words = num_words
        self.seq_length = seq_length
        self.language = language
        self.stopwords = stopwords
        self.specialwords = specialwords
        self.tfidf_vectorizer = TfidfVectorizer()

        logging.info('Clean and seg words {cleaned_corpus/seg_list} ...')
        self.cleaned_corpus = [self.cleaner(sent,
                                            pos=False,
                                            language=self.language,
                                            stopwords=self.stopwords,
                                            specialwords=self.specialwords,
                                            remove_alphas=remove_alphas,
                                            remove_numbers=remove_numbers,
                                            remove_urls=remove_urls,
                                            remove_punctuation=remove_punctuation,
                                            remove_ip_address=remove_ip_address) for sent in self.corpus]

        self.seg_list = self._seg_all(language=self.language, pos=False)
        if shuffle:
            self.seg_list = self.shuffle_data()

        logging.info('Build vocabulary {dictionary} ...')
        self.dictionary = corpora.Dictionary(self.seg_list)

        logging.info('Calculate tf-idf features {tf/idf/tfidf_vectorizer} ...')
        self.tf = self.get_tf(self.seg_list)
        self.idf = self.get_idf()
        self.tfidf_vectorizer.fit([" ".join(seg) for seg in self.seg_list])

        logging.info('Build tokenizer {tokenizer/sequences/word_index} ...')
        self.tokenizer = Spec_Tokenizer(num_words=self.num_words)
        self.tokenizer.fit_on_texts(self.seg_list)
        self.sequences = self.transform(reversed=False)
        self.word_index = self.tokenizer.word_index

        logging.info('Padding sequences {padded_sequences} ...')
        self.padded_sequences = self.padding()

        logging.info('load embedding matrix {embedding_matrix} ...')
        self.embedding_matrix = self.load_embedding_matrix(self.embedding_size, embedding_file_path, emb_type)

    def seg(self, sentence, pos, language):
        if language == 'zh':
            if not pos:
                seg_list = jieba.cut(sentence)
            else:
                seg_list = psg.cut(sentence)
            result = [item for item in seg_list if item != ' ']
        else:
            if not pos:
                result = sentence.split()
            else:
                result = nltk.pos_tag(sentence)
        return result

    def cleaner(self, text, pos, language, stopwords=None, specialwords=None, remove_alphas=False, remove_numbers=False,
                remove_urls=False, remove_punctuation=False, remove_ip_address=False):
        alphas = re.compile(r"[A-Za-z]+", re.IGNORECASE)
        numbers = re.compile(r"\d+", re.IGNORECASE)
        email = re.compile(r"", re.IGNORECASE)
        url_1 = re.compile(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)", re.IGNORECASE)
        url_2 = re.compile(r"[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)", re.IGNORECASE)
        url_3 = re.compile(r"(https?|ftp|file)://[-A-Za-z0-9+&&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", re.IGNORECASE)
        ips = re.compile(r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}", re.IGNORECASE)

        chinese_pattern = re.compile(u'[\u4e00-\u9fa5]+')
        stop_p = p + "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰�〾〿–—‘・’‛“”„‟…‧﹏."

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

        # 关键成分过滤
        text = re.sub(r'\u0020+', ' ', text)
        text = re.sub(r'\xa0+', ' ', text)
        if remove_alphas:
            text = alphas.sub(' ', text)
        if remove_numbers:
            text = numbers.sub(' ', text)
        if remove_urls:
            text = url_1.sub(' ', text)
            text = url_2.sub(' ', text)
            text = url_3.sub(' ', text)
        if remove_ip_address:
            text = ips.sub(' ', text)
        if remove_punctuation:
            for punc in stop_p:
                if punc in text:
                    text = text.replace(punc, ' ')
        else:
            if language == 'zh':
                text = re.sub(r'!', '！', text)
                text = re.sub(r',', '，', text)
                text = re.sub(r'\(', '（', text)
                text = re.sub(r'\)', '）', text)
                text = re.sub(r'\.', '。', text)
                text = re.sub(r':', '：', text)
                text = re.sub(r'\?', '？', text)
            else:
                text = re.sub(r'！', '!', text)
                text = re.sub(r'〝', '"', text)
                text = re.sub(r'〞', '"', text)
                text = re.sub(r'，', ',', text)
                text = re.sub(r'（', '(', text)
                text = re.sub(r'）', ')', text)
                text = re.sub(r'。', '.', text)
                text = re.sub(r'：', ':', text)
                text = re.sub(r'？', '?', text)

        # 英文规范
        if specialwords is not None:
            for word in specialwords:
                if word in text:
                    text = text.replace(word, word.capitalize())
        text = text.lower()

        # 去停用词
        if stopwords is not None:
            text = "".join([char for char in text if char not in stopwords])

        # 统一规范
        text = re.sub(r" +", " ", text)

        return text

    def get_by_postag(self, sentence, tag, language):
        seg_list = self.seg(sentence, pos=True, language=language)
        filter_list = []

        if language == 'zh':
            for seg in seg_list:
                word = seg.word
                flag = seg.flag

                if not flag.startswith(tag):
                    continue
                filter_list.append(word)
        else:
            for word, flag in seg_list:
                if not flag.startswith(tag):
                    continue
                filter_list.append(word)

        return filter_list

    def get_cossim(self, sent1, sent2):
        if isinstance(sent1, str):
            seg_sent1 = [" ".join(self.seg(sent1, pos=False))]
        else:
            seg_sent1 = [" ".join(self.seg(sent, pos=False)) for sent in sent1]
        if isinstance(sent2, str):
            seg_sent2 = [" ".join(self.seg(sent2, pos=False))]
        else:
            seg_sent2 = [" ".join(self.seg(sent, pos=False)) for sent in sent2]
        s1_matrix = self.tfidf_vectorizer.transform(seg_sent1)
        s2_matrix = self.tfidf_vectorizer.transform(seg_sent2)
        return cs(s1_matrix, s2_matrix).flatten()

    def get_info_entropy(self, array):
        if isinstance(array, list):
            array = np.array(array)

        x_value_list = np.unique(array)
        ent = 0.0
        for x_value in x_value_list:
            p = map(len(x[x == x_value]) / len(x), float)
            logp = np.log2(p)
            ent -= p * logp

        return ent

    def get_gini_coefficients(self, array):
        if isinstance(array, list):
            array = np.array(array)

        mad = np.abs(np.subtract.outer(array, array)).mean()
        rmad = mad / np.mean(array)
        g = 0.5 * rmad

        return g

    def word_count(self):
        word_dict = defaultdict(int)
        if not self.pos:
            for sent in self.seg_list:
                for word in sent:
                    word_dict[word] += 1
        word_dict = sorted(word_dict.items(), key=operator.itemgetter(1), reversed=True)
        return word_dict

    def get_tf(self, corpus):
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

    def get_idf(self):
        df_dic, idf_dic = defaultdict(int), defaultdict(int)
        tt_count = len(self.seg_list)

        for doc in self.seg_list:
            for word in set(doc):
                df_dic[word] += 1

        for k, v in df_dic.items():
            idf_dic[k] = math.log(tt_count) - math.log(v + 1)

        default_idf = math.log(tt_count / 1.0)

        return idf_dic, default_idf

    def get_bow(self):
        return [self.dictionary.doc2bow(doc) for doc in self.seg_list]

    def shuffle_data(self):
        indices = np.arange(len(self.seg_list))
        shuffled_indices = np.random.permutation(indices)
        return self.seg_list[shuffled_indices]

    def transform(self, reversed=False):
        if reversed:
            return self.tokenizer.sequences_to_texts(self.seg_list)
        else:
            return self.tokenizer.texts_to_sequences(self.seg_list)

    def padding(self):
        return pad_sequences(self.sequences, maxlen=self.seq_length)

    def make_batch(self):
        pass

    def get_stopwords(self, filename, encoding='utf-8'):
        return [sw.strip('\n') for sw in open(filename, 'r', encoding=encoding).readlines()]

    def load_embedding_matrix(self, embedding_size, embedding_file_path, type='self'):
        nb_words = len(self.dictionary)

        print('Creating embedding matrix ...')
        if embedding_file_path is not None:
            assert embedding_size is not None
            embedding_matrix = np.zeros((nb_words + 2, embedding_size))
            embedding_dict = {}
            if type == 'self':
                model = word2vec.Word2Vec.load(embedding_file_path)
                for word in model.wv.vocab.keys():
                    embedding_dict[word] = model.wv[word]
            else:
                with open(embedding_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        array = line.strip()
                        word = "".join(array[:-embedding_size])
                        vector = list(map(float, array[-embedding_size:]))
                        embedding_dict[word] = vector

            for idx, word in self.dictionary.items():
                if word in embedding_dict:
                    embedding_vector = embedding_dict.get(word)
                    embedding_matrix[idx] = embedding_vector
                else:
                    embedding_vector = np.random.uniform(low=-0.2, high=0.2, size=embedding_size)
                    embedding_matrix[idx] = embedding_vector
        else:
            embedding_matrix = np.random.uniform(low=-0.2, high=0.2, size=(nb_words + 2, embedding_size))

        return embedding_matrix

    def _seg_all(self, language, pos=False):
        return [self.seg(sent, pos, language) for sent in self.cleaned_corpus]


class algorithms(object):
    """
    Traditional machine/deep learning algorithms for nlp
    Include:
        - keyword extraction
        - text classification
    """
    def __init__(self, nlp_tools):
        self.nlp_tools = nlp_tools

        logging.info('**Tool Guides**')
        fun_names = [name for name in dir(self) if '__' not in name and '_' != name[0]]
        logging.info('This tool contains these features...')
        logging.info('function names:\n' + '\n'.join(fun_names))

    # 关键词提取(一句话粒度)
    def keyword_extraction(self, seg_list, keyword_num=2, algorithm='tfidf', stopword_file=None):
        if algorithm == 'tfidf':
            return self._tfidf_keyword_extractor(seg_list, keyword_num)

        elif algorithm == 'textrank':
            return self._textrank_keyword_extractor(seg_list, keyword_num, stopword_file)

        elif algorithm == 'lsi':
            return self._lsi_keyword_extractor(seg_list, keyword_num, num_topic=3)

        elif algorithm == 'lda':
            return self._lda_keyword_extractor(seg_list, keyword_num, num_topic=3)

        else:
            raise ValueError('The algorithm is not implemented!')

    def _tfidf_keyword_extractor(self, seg_list, keyword_num):
        tfidf_dic = {}
        tf_dic = self.nlp_tools.get_tf([seg_list])[0]
        idf_dic, default_idf = self.nlp_tools.idf

        for word in seg_list:
            idf = idf_dic.get(word, default_idf)
            tf = tf_dic.get(word, 0)
            tfidf = tf * idf
            tfidf_dic[word] = tfidf

        return [k for k, _ in sorted(tfidf_dic.items(), key=oprator.itemgetter(1), reverse=True)][:keyword_num]

    def _textrank_keyword_extractor(self, seg_list, keyword_num, stopword_file=None):
        if stopword_file is not None:
            analyse.set_stop_words(stopword_file)
        textrank = analyse.textrank

        return textrank(seg_list, keyword_num)

    def _lsi_keyword_extractor(self, seg_list, keyword_num, num_topic):
        corpus = self.nlp_tools.get_bow()
        dictionary = self.nlp_tools.dictionary

        tfidf_model = models.TfidfModel(corpus)
        corpus_tfidf = tfidf_model[corpus]
        model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topic)

        wordtopic_dic = self._get_word_topic_dict(dictionary, tfidf_model, model)
        return _get_topic_sim(seg_list, dictionary, tfidf_model, model, wordtopic_dic)[:keyword_num]

    def _lda_keyword_extractor(self, seg_list, keyword_num, num_topic):
        corpus = self.nlp_tools.get_bow()
        dictionary = self.nlp_tools.dictionary

        tfidf_model = models.TfidfModel(corpus)
        corpus_tfidf = tfidf_model[corpus]
        model = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=num_topic)

        wordtopic_dic = self._get_word_topic_dict(dictionary, tfidf_model, model)
        return _get_topic_sim(seg_list, dictionary, tfidf_model, model, wordtopic_dic)[:keyword_num]

    def _get_word_topic_dict(self, dictionary, tfidf_model, model):
        wordtopic_dic = {}
        word_dict = dictionary.token2id

        for word, _ in word_dict.items():
            single_word = [word]
            wordcorpus = tfidf_model[dictionary.doc2bow(single_word)]
            wordtopic = model[wordcorpus]
            wordtopic_dic[word] = wordtopic

        return wordtopic_dic

    def _get_topic_sim(self, seg_list, dictionary, tfidf_model, model, wordtopic_dic):
        sentcorpus = tfidf_model[dictionary.doc2bow(seg_list)]
        senttopic = model[sentcorpus]

        sim_dict = {}
        for word in seg_list:
            if word in wordtopic_dic:
                word_topic = wordtopic_dic[word]
                sim = cs([[item[1] for item in word_topic]], [[item[1] for item in senttopic]])
                sim_dict[word] = sim[0][0]

        return [k for k, _ in sorted(sim_dict.items(), key=operator.itemgetter(1), reverse=True)]

