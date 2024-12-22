import warnings
warnings.filterwarnings('ignore')
import pandas as pd
#显示所有列
from _operator import itemgetter
import json
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
import numpy as np


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
    return docs, stopwords, df['abstract'].tolist(),df



def train(docs,num_topic):
    """
    训练lda模型
    """
    dictionary = Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    lda_model = LdaModel(corpus = corpus, id2word = dictionary, num_topics = 10)

    return lda_model,dictionary,corpus



def test(lda_model, dictionary, corpus , docs , doc_index, label_list,top_n=10):
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
    index = similarities.MatrixSimilarity(lda_model[corpus])#把所有数据做成索引
    query = docs[doc_index]#可根据索引查询
    input_line = "东部海洋经济圈的海洋科技发展水平和海洋经济 。为科学合理评价区域海洋科技发展水平,进一步促进区域海洋经济增长,文章以东部海洋经济圈为例,采用主成分分析方法构建海洋科技发展水平综合指标,采用多元线性回归模型分析海洋科技发展等因素与海洋经济增长之间的关系,并提出建议。研究结果表明:海洋科技发展水平综合指标包括海洋科技支撑、海洋科技成果转化和海洋科技投入3个主要素及其11个指标,2006—2015年东部海洋经济圈各地海洋科技发展水平整体处于上升趋势,且差距较小;除政府支持力度外,海洋科技发展水平、海洋第三产业发展水平和工业污染程度等变量均与海洋生产总值存在显著相关关系,其中海洋科技发展水平为正相关;未来应促进海洋科技成果转化、减少区域壁垒制约以及加强海洋生态环境保护和治理。"
    print('测试文本', input_line)
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', input_line)
    word_list = [w for w in list(jieba.cut(line)) if w not in stopwords]
    query = word_list
    vec_bow = dictionary.doc2bow(query)#把数据变成词袋
    vec_model = lda_model[vec_bow]#将该词袋引用于LDA
    sims = index[vec_model]#利用索引计算词袋内向量的相似度
    sims = sorted(enumerate(sims), key=lambda item: -item[1])#把iterable中的items进行排序之后，返回一个新的列表；key=lambda item: -item[1]以值排序
    #print('测试文本是:', query)
    for index, doc in enumerate(sims[:10]):
        docs_index = doc[0]
        score = doc[1]
        print(index+1, raw_line[docs_index][:100], score, label_list[docs_index])
        #print(index+1,' '.join(docs[docs_index][:30]),score)
    return sims[:top_n]


def topic(df):
    """
    训练模型
    """
    # 原始语料集合
    corpus_lda = lda_model[corpus]
    f = open('../data/result.txt' , "w+")
    for line in lda_model.print_topics(num_topic, num_words=30):#打印出主题分布，由30个关键词组成
        f.write(str(line))
        f.write('\n')
    f.close()
    label_list = []
    topic_martix=np.zeros((len(corpus_lda),num_topic))#制作主题矩阵；np.zeros返回来一个给定形状和类型的用0填充的数组
    for i,ele in enumerate(corpus_lda):
        for num in ele:
            j=num[0]
            value=num[1]
            topic_martix[i][j]=value#
        label = max(ele, key=itemgetter(1))[0]#到最大值结束
        label_list.append(label)

    print(topic_martix)
    result_dict = {}
    for ele in label_list:#遍历得到数量
        if (ele not in result_dict):
            result_dict[ele] = 1#找到词就加一
        else:
            result_dict[ele] = result_dict[ele] + 1
    jsObj = json.dumps(result_dict)#把python对象转换成json对象的一个过程，生成的是字符串
    fileObject = open('../data/count.json', 'w')#主题下的文字数量
    fileObject.write(jsObj)
    fileObject.close()
    df['topic_probability']=topic_martix.tolist()#主题矩阵写入
    df.to_csv('../data/output_file.csv',index=None,encoding='utf-8-sig')
    return label_list



if __name__=="__main__":
    start_time = time.time()
    docs ,stopwords, raw_line , df= preprocess()
    num_topic = 10
    lda_model, dictionary, corpus = train(docs, num_topic)
    label_list = topic(df)
    doc_index = 2
    test(lda_model, dictionary, corpus, docs, doc_index,label_list)
    end_time = time.time()
    print('lda cost time is:', end_time - start_time)