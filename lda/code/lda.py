import warnings
warnings.filterwarnings('ignore')
import pandas as pd
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
import re
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import jieba
from gensim import similarities
import time

def preprocess():
    """
    去除特殊符号、停用词、分词、构造lda模型训练集
    :return:
    """
    df = pd.read_csv('../data/abstract.csv')
    def remove_punctuation(line):
        line = str(line)
        if line.strip() == '':
            return ''
        rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
        line = rule.sub('', line)
        return line
    def stopwordslist(filepath):
        stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
        return stopwords
    stopwords = stopwordslist("../data/stopwords.txt")
    df['clean_abstract'] = df['abstract'].apply(remove_punctuation)
    df['cut_abstract'] = df['clean_abstract'].apply(lambda x: [w for w in list(jieba.cut(x)) if w not in stopwords])
    #print(df.head())
    docs = df['cut_abstract'].tolist()
    return docs, stopwords, df['abstract'].tolist()



def train(docs):
    """
    训练lda模型
    """
    #start_time = time.time()
    dictionary = Dictionary(docs)#词典
    corpus = [dictionary.doc2bow(doc) for doc in docs]#向量
    lda_model = LdaModel(corpus = corpus, id2word = dictionary, num_topics = 10)
    #lda_model = LdaModel(corpus=corpus, id2word=dictionary, chunksize=2000, alpha='auto',eta='auto',iterations=400, passes=20,eval_every=None, num_topics=3)
    #end_time = time.time()
    #print('lda model training time is', end_time - start_time)
    return lda_model,dictionary,corpus



def test(lda_model, dictionary, corpus , docs , doc_index,  top_n=10):
    """
    预估计算相似性
    :param lda_model:
    :param dictionary:
    :param corpus:
    :param docs:
    :param doc_index:
    :param top_n:
    :return:
    """
    index = similarities.MatrixSimilarity(lda_model[corpus])
    #query = docs[doc_index]
    input_line = "东部海洋经济圈的海洋科技发展水平和海洋经济 。为科学合理评价区域海洋科技发展水平,进一步促进区域海洋经济增长,文章以东部海洋经济圈为例,采用主成分分析方法构建海洋科技发展水平综合指标,采用多元线性回归模型分析海洋科技发展等因素与海洋经济增长之间的关系,并提出建议。研究结果表明:海洋科技发展水平综合指标包括海洋科技支撑、海洋科技成果转化和海洋科技投入3个主要素及其11个指标,2006—2015年东部海洋经济圈各地海洋科技发展水平整体处于上升趋势,且差距较小;除政府支持力度外,海洋科技发展水平、海洋第三产业发展水平和工业污染程度等变量均与海洋生产总值存在显著相关关系,其中海洋科技发展水平为正相关;未来应促进海洋科技成果转化、减少区域壁垒制约以及加强海洋生态环境保护和治理。"
    print('测试文本', input_line)
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', input_line)
    word_list = [w for w in list(jieba.cut(line)) if w not in stopwords]
    query = word_list
    vec_bow = dictionary.doc2bow(query)#文档 query变成一个稀疏向量
    vec_model = lda_model[vec_bow]
    sims = index[vec_model]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    #print('测试文本是:', query)
    for index, doc in enumerate(sims[:10]):
        docs_index = doc[0]
        score = doc[1]
        print(index+1, raw_line[docs_index][:100], score)
        #print(index+1,' '.join(docs[docs_index][:30]),score)
    return sims[:top_n]




if __name__=="__main__":
    start_time = time.time()
    docs ,stopwords, raw_line = preprocess()
    lda_model, dictionary, corpus = train(docs)
    doc_index = 2
    test(lda_model, dictionary, corpus, docs, doc_index)
    end_time = time.time()
    print('lda cost time is:', end_time - start_time)