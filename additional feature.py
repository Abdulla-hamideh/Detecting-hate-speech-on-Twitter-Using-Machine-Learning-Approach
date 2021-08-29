import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import seaborn as sns
from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer as VS
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE


dataset = pd.read_csv("ready_dataset.csv")
tweet=dataset.clean_tweet

sentiment_analyzer = VS()


def count_tags(tweet_c):
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', tweet_c)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return (parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'), parsed_text.count('HASHTAGHERE'))


def sentiment_analysis(tweet):
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    twitter_objs = count_tags(tweet)
    features = [sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'], twitter_objs[0],
                twitter_objs[1],
                twitter_objs[2]]
    # features = pandas.DataFrame(features)
    return features


def sentiment_analysis_array(tweets):
    features = []
    for t in tweets:
        features.append(sentiment_analysis(t))
    return np.array(features)


final_features = sentiment_analysis_array(tweet)
# final_features

new_features = pd.DataFrame({'Neg': final_features[:, 0], 'Pos': final_features[:, 1], 'Neu': final_features[:, 2],
                                'Compound': final_features[:, 3],
                                'url_tag': final_features[:, 4], 'mention_tag': final_features[:, 5],
                                'hash_tag': final_features[:, 6]})


tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3),max_df=0.75, min_df=5, max_features=10000)

# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(dataset['clean_tweet'] )

tfidf_a = tfidf.toarray()
modelling_features = np.concatenate([tfidf_a,final_features],axis=1)

X = pd.DataFrame(modelling_features)
y = dataset['class'].astype(int)

X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# for the imbalance data
sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_resample(X_train_tfidf,y_train)


# running it over the model
def evaluate(X_train_res, X_test_tfidf, y_train_res, y_test):
    model_name_list_class = ["Logistic Regression", "Linear SVM",'Random Forest',"Bayes_Naive"]
    # Instantiate the models
    logistic_Regression = LogisticRegression(class_weight='balanced',penalty="l2",C=0.05,max_iter=100000)
    Linear_SVC = LinearSVC(class_weight='balanced',C=0.05, penalty='l2', loss='squared_hinge',multi_class='ovr', max_iter=10000)
    Random_forest = RandomForestClassifier()
    Bayes_Naive = DecisionTreeClassifier()
    # Dataframe for results
    results1 = pd.DataFrame(columns=['accuracy'], index=model_name_list_class)
    for i, model in enumerate([logistic_Regression, Linear_SVC, Random_forest,Bayes_Naive]):
        model.fit(X_train_res, y_train_res)
        y_preds = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_preds)
        report = classification_report(y_test, y_preds)
        print(report)
        print(model_name_list_class[i],":" , acc)
        confusion = confusion_matrix(y_test, y_preds)
        matrix_proportions = np.zeros((3, 3))
        for x in range(0, 3):
            matrix_proportions[x, :] = confusion[x, :] / float(confusion[x, :].sum())
        names = ['Hate', 'Offensive', 'Neither']
        confusion_df = pd.DataFrame(matrix_proportions, index=names, columns=names)
        plt.figure(figsize=(5, 5))
        sns.heatmap(confusion_df, annot=True, annot_kws={"size": 12}, cmap='YlGnBu', cbar=False, square=True, fmt='.2f')
        plt.ylabel(r'True Value', fontsize=14)
        plt.xlabel(r'Predicted Value', fontsize=14)
        plt.tick_params(labelsize=12)
        plt.title(model_name_list_class[i])
        plt.show()
        # Insert results into the dataframe
        model_name = model_name_list_class[i]
        results1.loc[model_name, :] = [acc]
    return results1

# looking at the dataframe
results_withSMOTE_class = evaluate(X_train_res, X_test_tfidf, y_train_res, y_test)
print(results_withSMOTE_class )
