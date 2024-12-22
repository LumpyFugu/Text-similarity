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
import jieba
import gensim
from gensim.models.doc2vec import Doc2Vec
import time#用来记时
TaggededDocument = gensim.models.doc2vec.TaggedDocument


def preprocess():
    """
    去除特殊符号、停用词、分词、构造训练集合
    :return:
    """
    df = pd.read_csv('../data/abstract.csv')#读取文件
    def remove_punctuation(line):#移除标点符号
        line = str(line)
        if line.strip() == '':#移除字符串头尾的空格
            return ''
        rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")#由数字、26 个英文字母或者下划线组成的字符串
        line = rule.sub('', line)
        return line
    def stopwordslist(filepath):
        stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
        return stopwords
    stopwords = stopwordslist("../data/stopwords.txt")
    df['clean_abstract'] = df['abstract'].apply(remove_punctuation)
    df['cut_abstract'] = df['clean_abstract'].apply(lambda x: [w for w in list(jieba.cut(x)) if w not in stopwords])
    #print(df.head())
    docs = df['cut_abstract'].tolist()#将数组作为（可能是嵌套的）列表返回
    x_train = []
    # y = np.concatenate(np.ones(len(docs)))
    for i, word_list in enumerate(docs):#将一个可遍历的数据对象(列表)组合为一个索引序列，同时列出数据和数据下标
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip()
        document = TaggededDocument(word_list, tags=[i])#为文本打标签，以供doc2vec使用
        x_train.append(document)
    return x_train,docs,stopwords,df['abstract'].tolist()


def train(x_train, size=200, epoch_num=70):
    """
    训练doc模型
    :param x_train:
    :param size:
    :param epoch_num:
    :return:
    """

    model_doc2vec= Doc2Vec(x_train, min_count=1, window=3, vector_size=200, negative=5, workers=4)
    model_doc2vec.train(x_train, total_examples=model_doc2vec.corpus_count, epochs=70)
    model_doc2vec.save('../model/model_doc2vec')



def test(docs, index):
    model_doc2vec = Doc2Vec.load("../model/model_doc2vec")
    input_line = "东部海洋经济圈的海洋科技发展水平和海洋经济 。为科学合理评价区域海洋科技发展水平,进一步促进区域海洋经济增长,文章以东部海洋经济圈为例,采用主成分分析方法构建海洋科技发展水平综合指标,采用多元线性回归模型分析海洋科技发展等因素与海洋经济增长之间的关系,并提出建议。研究结果表明:海洋科技发展水平综合指标包括海洋科技支撑、海洋科技成果转化和海洋科技投入3个主要素及其11个指标,2006—2015年东部海洋经济圈各地海洋科技发展水平整体处于上升趋势,且差距较小;除政府支持力度外,海洋科技发展水平、海洋第三产业发展水平和工业污染程度等变量均与海洋生产总值存在显著相关关系,其中海洋科技发展水平为正相关;未来应促进海洋科技成果转化、减少区域壁垒制约以及加强海洋生态环境保护和治理。"
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', input_line)
    word_list = [w for w in list(jieba.cut(line)) if w not in stopwords]
    print('测试文本是:', input_line)#输入测试文本并同样进行预处理的操作

    # 句子向量化操作
    inferred_vector_dm = model_doc2vec.infer_vector(word_list)#通过 infer_vector函数将句子向量化
    sims = model_doc2vec.docvecs.most_similar([inferred_vector_dm], topn=10)#使用most_similar计算相似度
    count_num = 1
    for count, sim in sims:#按序输出
        sentence = x_train[count]
        #print(sentence, 11111)
        print(count_num, raw_lines[count][:100], sim)
        #print(count_num, ' '.join(sentence[0][:30]), sim)
        count_num += 1
    return sims




if __name__=="__main__":
    start_time = time.time()
    x_train,docs, stopwords ,raw_lines= preprocess()
    train(x_train)
    index = 2
    test(docs, index)
    end_time = time.time()
    print('doc2vec cost time is:', end_time - start_time)
