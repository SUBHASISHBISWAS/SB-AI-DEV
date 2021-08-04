import string
from nltk.corpus import stopwords
def textProcessing(data):
    #Remove Punctuations
    removePunctuation = [character for character in data if character not in string.punctuation]
    #Convert Chars to Sentence
    sentenceWithoutPunctuations = ''.join(removePunctuation)
    words = sentenceWithoutPunctuations.split()
    #Remove Stopwords
    removeStopwords = [word for word in words if word.lower() not in stopwords.words('english')]
    #Return final list of words
    return removeStopwords