
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

df_fake["class"] = 0
df_true["class"] = 1

#Unión del Dataset de noticias Falsas con el de las noticias Reales
df_merge = pd.concat([df_fake, df_true], axis =0 )
df_merge.columns

#Se suprimen columnas innecesarias a la hora del entrenamiento
df = df_merge.drop(["text","subject","date"], axis = 1)

#Una vez unidas las noticias de ambos Datasets, se mezclan aleatoriamente
df = df.sample(frac = 1)
print(df.head())

#Se establece un índice para una mejor organización
df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)
print(df.columns)
print(df.head())

##################################################

#Función que procesa los textos de las noticias y las prepara para el entrenamiento
def wordopt(title):
    title = title.lower()
    title = re.sub('\[.*?\]', '', title)
    title = re.sub("\\W"," ",title) 
    title = re.sub('https?://\S+|www\.\S+', '', title)
    title = re.sub('<.*?>+', '', title)
    title = re.sub('[%s]' % re.escape(string.punctuation), '', title)
    title = re.sub('\n', '', title)
    title = re.sub('\w*\d\w*', '', title)    
    return title
df["title"] = df["title"].apply(wordopt)

#Variables
x = df["title"]
y = df["class"]

#Sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

#Conversión del texto en vectores
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


#################################################################################3

from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Crear un clasificador SVM (Support Vector Machine)
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(xv_train, y_train)

# Predicciones CON el conjunto de prueba
pred_svm = svm_classifier.predict(xv_test)

# Evaluación del rendimiento del modelo
accuracy_svm = svm_classifier.score(xv_test, y_test)
classification_report_svm = classification_report(y_test, pred_svm)

print(f'Precisión (SVM): {accuracy_svm}')
print(f'Classification Report (SVM):\n{classification_report_svm}')

#LOGISTIC REGRESION (LR)
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(max_iter=10000000)
LR.fit(xv_train,y_train)

pred_lr=LR.predict(xv_test)
print(LR.score(xv_test, y_test))
print(classification_report(y_test, pred_lr))


#DECISION TREE (DT)
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

pred_dt = DT.predict(xv_test)
print(DT.score(xv_test, y_test))
print(classification_report(y_test, pred_dt))


#GRADIENT BOOSITNG CLASSIFIER (GBC)
from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)

pred_gbc = GBC.predict(xv_test)
print(GBC.score(xv_test, y_test))
print(classification_report(y_test, pred_gbc))

#RANDOM FOREST CLASSIFIER (RFC)
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)

pred_rfc = RFC.predict(xv_test)
print(RFC.score(xv_test, y_test))
print(classification_report(y_test, pred_rfc))

###########################################################################

# Test
def output_lable(n):
    if n == 0:
        return "Fake New"
    elif n == 1:
        return "Noticia Real"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return print("\n\nLR Predicción: {} \nDT Predicción: {} \nGBC Predicción: {} \nRFC Predicción: {}".format(output_lable(pred_LR[0]),
                                                                                                              output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_GBC[0]), 
                                                                                                              output_lable(pred_RFC[0])))


news = str(input("\n\n\nIntroduce el texto de la noticia:"))
manual_testing(news)

