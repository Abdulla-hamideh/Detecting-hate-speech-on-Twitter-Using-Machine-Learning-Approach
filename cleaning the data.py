# importing the necessary libraries
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')



# reading the file
data = pd.read_csv("labeled_data.csv")

# looking at the data
#print(data.head(10).to_string())

# deleting the unnamed coloumn
del data["Unnamed: 0"]

#cleaning the text
tweet = data["tweet"]
clean_tweet = []

# replacing the words
for index, row in data.iterrows():
    line = str(row['tweet']).lower()
    clean_tweet.append(' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(rt)"," ",line).split()))

# adding it to the dataframe
data["clean_tweet"] = clean_tweet

# #Lemmitization
lemmatizer = WordNetLemmatizer()
data["clean_tweet"] = data["clean_tweet"].apply(lambda x : ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

print(data.head(10).to_string())

# convert into a new csv
#data.to_csv("clean_tweets.csv", index=False)
