import nltk
import numpy as np
import random
import string # to process standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
f=open('google_info.txt','r',errors = 'ignore')
raw=f.read()
raw=raw.lower()# converts to lowercase

sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words



lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
# print (string.punctuation)
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
# print (remove_punct_dict)
def LemNormalize(text):
    # print (text)
    # print (text.lower().translate(remove_punct_dict))
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def response(user_response):
    robo_response=''
    # print (sent_tokens)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    # print (TfidfVec.fit(sent_tokens))
    tfidf = TfidfVec.fit_transform(sent_tokens)
    # print (TfidfVec.get_feature_names())
    # print (tfidf)
    # print(list(tfidf.toarray()))
    vals = cosine_similarity(tfidf[-1], tfidf)
    # print (vals)
    idx=vals.argsort()[0][-2]
    # print (vals.argsort())
    flat = vals.flatten()
    # print (flat)
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

flag=True
print("ROBO: My name is Robo. I will answer your queries about google. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("ROBO: "+greeting(user_response))
            else:
                sent_tokens.append(user_response)
                word_tokens=word_tokens+nltk.word_tokenize(user_response)
                # print ("&&&&",word_tokens)
                final_words=list(set(word_tokens))
                print("ROBO: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("ROBO: Bye! take care..")