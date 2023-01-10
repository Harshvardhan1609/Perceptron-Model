import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import plotly.express as px
from sklearn.linear_model import Perceptron as ps
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix


st.title("Perceptron Model")
st.sidebar.title("Values to Vary")
name = st.text_input("Name of database ", "diabetes")

my_data = pd.read_csv(f'{name}.csv')
# st.dataframe(my_data)

y = my_data.tail()
y = my_data["Outcome"]

features  = st.sidebar.slider("No of features" , min_value=2 , max_value=my_data.shape[1]-1,)
randomstate = 42
randomstate = st.sidebar.number_input('Please enter the random state of model 🔥')
testsize = st.sidebar.slider("test_size" , min_value=0.20 , max_value=1.0)
X = my_data.iloc[:,[0,features]].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=testsize,random_state=35)

perception = ps(random_state=int(randomstate))
perception.fit(X_train,y_train)


y_pred = perception.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()
As = accuracy_score(y_test,y_pred)
st.write('Accuracy score : ' , As.round(2))
fig = px.density_heatmap(my_data)
con = px.box(my_data)
cons  = px.imshow(cm)
consts  = px.scatter(my_data)
consts2  = px.line(my_data)
consts3  = px.strip(my_data)



tab1, tab2 , tab3 , tab4 , tab5 , tab6 = st.tabs(["HeatMap" , "Box Plot" , "Confusion Matrix" , "Scatter PLot" , "line chart" , "Strip Plot"])
with tab1:
    st.plotly_chart(fig,theme="streamlit",use_container_width=True)
with tab2:
    st.plotly_chart(con,theme="streamlit",use_container_width=True)
with tab3:
    st.plotly_chart(cons,theme="streamlit",use_container_width=True)
with tab4:
    st.plotly_chart(consts,theme="streamlit",use_container_width=True)
with tab5:
    st.plotly_chart(consts2,theme="streamlit",use_container_width=True)
with tab6:
    st.plotly_chart(consts3,theme="streamlit",use_container_width=True)

