# importing the necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import TweetTokenizer


data = pd.read_csv("ready_dataset.csv")

# term frequncy-inverse
tok = TweetTokenizer( preserve_case=False,reduce_len=True,strip_handles=True,)

tfidf_vectorizer = TfidfVectorizer(tokenizer= tok.tokenize,
    lowercase=True,
    ngram_range=(1,3),
    stop_words="english",
    use_idf=False,
    smooth_idf=False,
    token_pattern=r'\w{2,}',
    norm=None,
    max_features=5000,
    sublinear_tf=True,
    strip_accents="unicode")


# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(data['clean_tweet'])
X = tfidf
y = data['class'].astype(int)
#print(y.value_counts())
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
results_withSMOTE_class = evaluate(X_train_res, X_test_tfidf, y_train_res, y_test)
print(results_withSMOTE_class )


