import bs4
import urllib.request
import pandas as pd
import re
import nltk
import matplotlib
import sklearn.model_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

%matplotlib inline

home = 'http://mlg.ucd.ie/modules/yalp'
hm_link = 'http://mlg.ucd.ie/modules/yalp/health_medical_list.html' #Health & Medical Reviews
aut_link = 'http://mlg.ucd.ie/modules/yalp/automotive_list.html'#Automotive Reviews

def read_bsns(link):    # Function to read link values for all the Bussiness inside a Catregory
    response = urllib.request.urlopen(link)
    html = response.read().decode()
    #print(html)
    soup = bs4.BeautifulSoup(html,"html.parser")
    #print(parser.prettify())

    bsns_list = list()
    for link in soup.find_all('a'):
        text = link.get('href')
        bsns_list.append(text)

    print(len(bsns_list),'business links have been read from', soup.find('h3','info').get_text())
    return bsns_list
    

def read_review(bsns_list):     #Function to read review text and rating stars and store it as CSV
    dataset = {}
    dataset['rvw_label'] = list()
    dataset['rvw_text'] = list()
    for est in bsns_list:
        response = urllib.request.urlopen(home+"/"+est)
        html1 = response.read().decode()
        soup = bs4.BeautifulSoup(html1,"html.parser")
        #print(soup.prettify())

        for review in soup.find_all('div','review'):
            try:
                # Store '1' if reviews are Positive(5&4 stars) and '0' for Negative(3,2&1 Stars)
                star_text = review.find('img').get('alt')
                if star_text in ['1-star', '2-star', '3-star']:
                    dataset['rvw_label'].append(0)
                elif star_text in ['5-star', '4-star']:
                    dataset['rvw_label'].append(1)
                else:
                    print('Flag', review)
                dataset['rvw_text'].append(review.find('p', 'text').get_text())
            except:
                print('Error!! in storing review by', review.find('i').get_text(), "on", soup.find('h3', 'info').get_text() )
                
    print(len(dataset['rvw_text']), "reviews have been read")                
    return dataset
    
#Task 1: Scraping reviews and storing along with binary class labels
#Categories selected: Health and Medical, Automotive Reviews


hm_list = read_bsns(hm_link)
aut_list = read_bsns(aut_link)
​
hm_dataset = read_review(hm_list)
aut_dataset = read_review(aut_list)
​
hm_dataframe = pd.DataFrame(hm_dataset)
aut_dataframe = pd.DataFrame(aut_dataset)
​
print("Storing reviews into CSV files: H&M.csv, Aut.csv")
hm_dataframe.to_csv('H&M.csv')
aut_dataframe.to_csv('Aut.csv')

#Test Pre-Processing and Tockenization

# define the function with NLTK Porter Stemming
def stem_tokenizer(text):
    # use the standard scikit-learn tokenizer first
    standard_tokenizer = CountVectorizer().build_tokenizer()
    tokens = standard_tokenizer(text)
    # then use NLTK to perform stemming on each token
    stemmer = PorterStemmer()
    stems = []
    for token in tokens:
        stems.append( stemmer.stem(token) )
    return stems
    
StopWord = stopwords.words('english')
print(len(StopWord), "stop words from NLTK corpus")
#Updating Stopword collection to accomodate stemmed stop words. The words are taken from the warning showed during CountVectorization
for i in ['abov', 'afterward', 'alon', 'alreadi', 'alway', 'ani', 'anoth', 'anyon', 'anyth', 'anywher', 'becam', 'becaus', 'becom', 'befor', 'besid', 'cri', 'describ', 'dure', 'els', 'elsewher', 'empti', 'everi', 'everyon', 'everyth', 'everywher', 'fifti', 'formerli', 'forti', 'ha', 'henc', 'hereaft', 'herebi', 'hi', 'howev', 'hundr', 'inde', 'latterli', 'mani', 'meanwhil', 'moreov', 'mostli', 'nobodi', 'noon', 'noth', 'nowher', 'onc', 'onli', 'otherwis', 'ourselv', 'perhap', 'pleas', 'seriou', 'sever', 'sinc', 'sincer', 'sixti', 'someon', 'someth', 'sometim', 'somewher', 'themselv', 'thenc', 'thereaft', 'therebi', 'therefor', 'thi', 'thu', 'togeth', 'twelv', 'twenti', 'veri', 'wa', 'whatev', 'whenc', 'whenev', 'wherea', 'whereaft', 'wherebi', 'wherev', 'whi', 'yourselv','anywh', 'becau', 'doe', 'el', 'elsewh', 'everywh', 'ind', 'otherwi', 'plea', 'somewh']:
    StopWord.append(i)
print(len(StopWord), "stop words after addition of stemmed stop words")

def vect(text_list):
    for i in range(len(text_list)):
        text_list[i] = text_list[i].lower()
        text_list[i] = re.sub(r'[0-9]+', '', text_list[i]) #To remove numbers(Year, Time, Fee) from the reviews
        
    #Count Vectorizer with Unigram and Bigram tokenzation. Stem Tokenization using PorterStemming
    #Ignore terms that have a document frequency strictly lower than 3 (min_df = 3)
    vectorizer = CountVectorizer(ngram_range = (1,2),stop_words = StopWord ,min_df = 5,tokenizer=stem_tokenizer)
    X = vectorizer.fit_transform(text_list)
    print(X.shape)
    #term = vectorizer.get_feature_names()
    #print(term)
    return X
    
  complete_dataset = hm_dataset['rvw_text']+aut_dataset['rvw_text']
#print(len(complete_dataset)) #2905
#print(complete_dataset[1450]) #First review of Automotive. Reviews are in Order

#Vectorising the complete dataset with Reviews from two categories
#X_full[0:1450] =  Health and Medicine review vectors
#X_full[1450:2905] = Automotive review vectors
X_full = vect(complete_dataset)
X_full.shape # Reviews are merged to anable cross model validation
