import re
import jieba

stopwords_path = ''
stopwords = set([w.strip('\n') for w in open(stopwords_path, 'r').readlines()])

def cut_words_sents(content):
    sentences = re.split(r'(\.|\!|\?|。|！|？|\.{6})', content)
    sentences
    words = []
    for sent in sentences:
        temp_words = jieba.lcut(sent)
        temp_words = [w for w in temp_words if w not in stopwords]
        words.extend(temp_words)
    return sentences, words
    

