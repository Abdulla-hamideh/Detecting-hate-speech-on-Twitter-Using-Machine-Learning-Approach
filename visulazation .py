# importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
# to get the sentiment score for the tweet
nltk.download('vader_lexicon')
## to get the stop words
nltk.download("stopwords")
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# reading the file
data =pd.read_csv("clean_tweets.csv")
print(data.head(10).to_string())

# descriptive statistics
print(data.info())
print(data.describe().to_string())

# plot the class
# count we have 0s and 1s and 2s
modified = data.groupby("class")["class"].count()
sns.set(style="darkgrid")
# xlabel
x= ["Hate","Offensive Language","Neither"]
total = data["class"].count()
# plot
ax= sns.barplot(x=x,y=modified,edgecolor='black',palette="flare")
#plotting percentages on the bar
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width() / 2 - 0.05
    y = p.get_y() + p.get_height()
    ax.annotate(percentage, (x, y), size=12)
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# plotting the key bad word
hate_words = ' '.join([word for word in data['clean_tweet'][data['class'] == 0]])
wordcloud = WordCloud(max_font_size = 110,max_words = 100,background_color="black",colormap="Reds").generate(hate_words)
plt.axis('off')
plt.imshow(wordcloud, interpolation = 'bilinear',cmap='viridis')
plt.show()

# plotting hate speech
off_words = ' '.join([str(word) for word in data['clean_tweet'][data['class'] == 1]])
wordcloud = WordCloud(max_font_size = 110,max_words = 100,background_color="black",colormap="Reds").generate(off_words)
plt.axis('off')
plt.imshow(wordcloud, interpolation = 'bilinear',cmap='viridis')
plt.show()

# plotting offensive speech
normal_words = ' '.join([str(word) for word in data['clean_tweet'][data['class'] == 2]])
wordcloud = WordCloud(max_font_size = 110,max_words = 100, background_color="black",colormap="Reds").generate(normal_words)
plt.axis('off')
plt.imshow(wordcloud, interpolation = 'bilinear',cmap='viridis')
plt.show()

# plotting the frequency for the most used words in the sentences
text = ' '.join([str(word) for word in data['clean_tweet']])
tokenized_word=word_tokenize(text)
stop_words=set(stopwords.words("english"))

#adding more stop words
stop_words |= set(["lol",'@', '.',"...","..","I","'","n't","like","128514","'s","?","'m","got","get","8220","u","8230","You","know","na","The","This","If","128557","ca","'re","ai","yo","go","want",",","8221","see","would","My","one"])

# filtering more words in the sentence
filtered_sent=[]
for i in tokenized_word:
    if i not in stop_words:
        filtered_sent.append(i)

fdist = FreqDist(filtered_sent)
fdist.plot(25,cumulative=False)
plt.show()

# plotting the length of each class
modified_data = data
# measuring the length 
lenght = []
for w in modified_data["clean_tweet"]:
    oo = len(str(w))
    lenght.append(oo)
modified_data["length"] = lenght
#  changing the format of class variable
modified_data['class'] = modified_data['class'].map({0:'hate',1:'offensive',2:'neither'})
boxplot= sns.boxplot(x= "class",y= "length",data= modified_data,palette="flare",showfliers=False)
plt.show()

# skewness of the data in each class
graph = sns.FacetGrid(data=modified_data, col='class')
graph.map(plt.hist, 'length', bins=50)
plt.show()

# sentiment score
sid = SentimentIntensityAnalyzer()
modified_data['scores'] = modified_data['clean_tweet'].apply(lambda clean_tweet: sid.polarity_scores(str(clean_tweet)))
modified_data['Sentiment_Score'] = modified_data['scores'].apply(lambda score_dict: score_dict['compound'])
ax = sns.violinplot(x="class", y="Sentiment_Score", data=modified_data,palette="flare",inner="quartile")
plt.show()

# returning the actual labels for the original names
data["class"]=data['class'].map({'hate':0,'offensive':1,'neither':2})

# convert modified data into a new csv for modelling
modified_data.to_csv("ready_dataset.csv", index=False)
