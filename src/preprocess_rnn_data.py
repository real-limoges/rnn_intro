import csv
import nltk
import itertools

_VOCAB_SIZE = 8000
_UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'
_SENTENCE_START = 'SENTENCE_START'
_SENTENCE_END = 'SENTENCE_END'


def read_data(filename):
  with open(filename, 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)

    #Throw away header
    reader.next()

    sentences = itertools.chain(*[nltk.sent_tokenize(
      x[0].decode('utf-8').lower()) for x in reader])

    sentences = ["%s %s %s" % (_SENTENCE_START, x, _SENTENCE_END) 
                 for x in sentences]

  return sentences

def clean_data(data):
  tok_data = [nltk.word_tokenize(s) for s in data]

  word_freq = nltk.FreqDist(itertools.chain(*tok_data))

  vocab = word_freq.most_common(_VOCAB_SIZE-1)
  index_to_word = [word[0] for word in vocab]

  index_to_word.append(_UNKNOWN_TOKEN)

  word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

if __name__ == '__main__':
  data = read_data('../data/reddit-comments-2015-08.csv')
  clean_data(data)
