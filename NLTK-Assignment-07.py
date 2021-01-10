import nltk
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer as ps
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

#----- TASK 01 -----#
list_one = ["first_name", "last_name", "age", "occupation"]
some_tuple = ("John", "Holloway", 35, "carpenter")
print("LIST AND TUPLE INTO DICT: \n", dict(zip(list_one, some_tuple)))

#----- TASK 02 -----#
data = "All work and no play makes jack a dull boy, all work and no play"
print("\nWORD LEVEL TOKENIZATION: \n", word_tokenize(data))

data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
print("\nWORD LEVEL TOKENIZATION: \n", word_tokenize(data))
print("\nSENTENCE LEVEL TOKENIZATION: \n",sent_tokenize(data))

#----- TASK 03 -----#
stopWords = set(stopwords.words("english"))
words = word_tokenize(data.lower())
wordsFiltered = [w for w in words if w not in stopWords]

print("\nNumber of stopwords: \n", len(stopWords))
print("\nList of stopwords \n ", stopWords)
print("\nFiltered text \n ", wordsFiltered)

#----- TASK 04 -----#
words = ["game","gaming","gamed","games"]

print("\nStem Words:")
for word in words: print(ps().stem(word))

print("\nStem Sentence:")
sentence = "gaming, the gamers play games"
words = word_tokenize(sentence)
for word in words: print(ps().stem(word))

#----- TASK 05 -----#
sentence = "This is my sentence and I want to ngramize it."
n = 6
w_6grams = ngrams(sentence.split(), n)
print("\nWORD N-GRAMIZATION: \n")
for grams in w_6grams: print(grams)

c_6grams = ngrams(sentence, n)
print("\n CHAR N-GRAMIZATION: \n")
for grams in c_6grams: print("".join(grams))