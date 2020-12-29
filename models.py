import json
import sys
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv1D, MaxPooling1D
from tensorflow.python.keras.models import model_from_json

from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score

import pickle
import nltk

nltk.download('punkt')
nltk.download('stopwords')


def rem_stopwords(df_list):
    clean_list = []
    for row in df_list:
        clean_list.append([word for word in word_tokenize(row.lower()) if word not in stopwords.words('english')])
    return clean_list


def listtostring(lists):
    ans = ""
    for i in lists:
        for j in i:
            if j != " ":
                ans += str(j) + " "
    ans = ans[:-1]
    return ans



def tfidfer(text_from_df, vectorizer):
    corpus = [rows for rows in text_from_df]
    vectorizer.fit(corpus)


def transform(text, vectorizer):
    corpus = [rows for rows in text]
    vectors = vectorizer.transform(corpus)
    vectors = vectors.toarray()
    return vectors


def predict(headline, body):
    # test remove stopwords
    temp=headline.split(' ')
    test_clean_headlines = rem_stopwords(temp)
    test_clean_bodies = rem_stopwords(body.split(' '))
    print("removed test stopwords")

    test_clean_headlines = listtostring(test_clean_headlines)
    test_clean_bodies = listtostring(test_clean_bodies)


    # converting test lists to dataframe
    test_head_df = pd.DataFrame({'Body_ID': [1]})
    test_head_df['Headline'] = test_clean_headlines
    test_body_df = pd.DataFrame({'Body_ID': [1]})
    test_body_df['articleBody'] = test_clean_bodies

    combined_test = pd.merge(test_head_df, test_body_df, on='Body_ID')

    with open("test.txt", "r") as fp:
        merge_train = json.load(fp)

    merge2 = [row[0] + row[1] for row in merge_train]

    # tfidf fit
    print("Training tf-idf")
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidfer(merge2,vectorizer)

    # tfidf transform
    print("transforming tf-idf")
    test_headlines_vector = transform(combined_test.Headline,vectorizer)
    test_bodies_vector = transform(combined_test.articleBody,vectorizer)
    print("transforming complete")

    # combine test headlines and bodies tfidf vectors
    test_arr_combined = np.column_stack((test_headlines_vector, test_bodies_vector))

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")

    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    x_test = np.expand_dims(test_arr_combined, axis=2)
    preds = loaded_model.predict(x_test)

    return preds


# predict(headline='this is test headline', body='this  asdfas asdf asdf asdf is a body')
