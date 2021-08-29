import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk import TweetTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

data = pd.read_csv("ready_dataset.csv")

tok = TweetTokenizer(preserve_case=False,reduce_len=True,strip_handles=True,)

# lda
char_vectoris = CountVectorizer()
word_vec = CountVectorizer()
lda = LatentDirichletAllocation()


# TF-IDF feature matrix
char_vectorise= char_vectoris.fit_transform(data["clean_tweet"])
word_vecto = word_vec.fit_transform(data["clean_tweet"])
all_vect = hstack([word_vecto, char_vectorise])
lda_feature = lda.fit_transform(all_vect)
X = lda_feature
y = data['class'].astype(int)

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_resample(X, y)
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_train_res, y_train_res, random_state=0)

def evaluate(X_train_res, X_test_tfidf, y_train_res, y_test):
    model_name_list_class = ["Logistic Regression", "Linear SVM",'Random Forest',"Bayes_Naive"]
    # Instantiate the models
    logistic_Regression = LogisticRegression(max_iter=100000)
    Linear_SVC = LinearSVC()
    Random_forest = RandomForestClassifier()
    Bayes_Naive = MultinomialNB()
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
results_withSMOTE_class = evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test)
print(results_withSMOTE_class )