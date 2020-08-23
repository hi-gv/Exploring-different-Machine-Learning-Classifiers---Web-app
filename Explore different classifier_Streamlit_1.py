import streamlit as st 
import numpy as np 
from sklearn import datasets
#
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

st.title('Gaurav\'s Streamlit')


## adding text --markdown is used. '#' ~ H1 tag
st.write("""
# Explore different classifiers 
Which one is the best ?
""")

## select box 
# dataset_name = st.selectbox("Select Dataset",("Iris","Breast Cancer","Wine Dataset"))
#st.write(dataset_name)


## moving the abiove select box to sidebar
dataset_name = st.sidebar.selectbox("Select Dataset",("Iris","Breast Cancer","Wine Dataset"))
classifier_name = st.sidebar.selectbox("Select Classifier",("KNN","SVM","Random Forest"))


## fetching the dataset
def get_dataset(dataset_name):
    """
    Based on dataset we'll choose the right algos
    """
    if dataset_name =='Iris':
        data = datasets.load_iris()
    elif dataset_name =="Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target 
    return X,y
    
X, y = get_dataset(dataset_name)
st.write("Shape of dataset ", X.shape)
st.write("number of classes", len(np.unique(y)))  ## number of different classes in dataset


## based on classfiers we will output different parameters
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K",1,15)  ## slider widget into the sidebar
        params["K"] = K

    elif clf_name == "SVM":
        C = st.sidebar.slider("C",0.01,10.0)
        params["C"] = C 

    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    
    return params


params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
   
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors = params["K"])
       
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
     

    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth = params["max_depth"], random_state=1234)
    
    return clf

clf = get_classifier(classifier_name, params)

 ## Classification
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test,y_pred)
st.write(f"classifier = {classifier_name}")
st.write(f"accuracy={acc}")


## Plot
pca = PCA(2) ## no fo dimensions we need = 2
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0] ## all samples from dimension -'0'
x2 = X_projected[:, 1] ## all samples from dimension -'1'

fig = plt.figure()
plt.scatter(x1,x2,c=y, alpha=0.8, cmap="viridis") ## color - trasnparency - colormap
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

#plt.show() ## instead of this we'll sue below
st.pyplot()