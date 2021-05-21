# importing the necessary libraries
import pandas as pd
import re

# reading the file
data = pd.read_csv("labeled_data.csv")

# looking at the data
print(data.head(10).to_string())

# deleting the unnamed coloumn
del data["Unnamed: 0"]

#cleaning the text
tweet = data["tweet"]
clean_tweet = []
remove = ["#","&",'"',";","!",":","/","http","~"]

# replacing the words
for i in tweet:
    line = i[:i.find("/")]
    for g in remove:
        line = re.sub(g," ",line)
    line = re.sub("RT","",line)
    clean_tweet.append(line)

# adding it to the dataframe
data["clean_tweet"] = clean_tweet

# convert into a new csv
data.to_csv("clean_tweets.csv", index=False)
