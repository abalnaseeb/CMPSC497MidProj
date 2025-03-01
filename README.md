# News Classification
## Required Python packages
1. **[NumPy](https://numpy.org/doc/stable/)**
2. **[Pandas](https://pandas.pydata.org/docs/)**
3. **[TensorFlow](https://www.tensorflow.org/api_docs)**
4. **[Scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)**
5. **[Matplotlib](https://matplotlib.org/stable/contents.html)**
6. **[Wordcloud](https://pypi.org/project/wordcloud/)**
7. **[Keras-tuner](https://keras.io/keras_tuner/)**
8. **[Streamlit](https://docs.streamlit.io/)**
9. **[Psutil](https://psutil.readthedocs.io/en/latest/)**

## Installation
- Download Python3 from [Python Official Site](https://www.python.org/downloads/) and install it
- Download and install Jupyter notebook [jupyter.org](https://jupyter.org/install)
- Download and install pip: [https://pypi.org/project/pip/](https://pypi.org/project/pip/)
- Open terminal and go to project folder and run following command to install packages

  ```pip install -r requirements.txt```


## **Dataset**

1 - Download the GloVe embeddings from the following link (since github cannot store data bigger than 100MB):

 	https://nlp.stanford.edu/data/glove.6B.zip

2 - Extract the Files
After downloading the zip file make sure you unzip it, you should have the following .txt files:
- glove.6B.50d.txt
- glove.6B.100d.txt
- glove.6B.200d.txt
- glove.6B.300d.txt

  
3 - Organize the Files
Make sure those files are in project directory under `.../CMPSC497MidProj/Data/glove.6B`

![Screenshot 2025-02-28 at 7 35 59 PM](https://github.com/user-attachments/assets/d754fc18-c7ca-473f-9603-47bd3f2a9b4c)

## Run Application
-  Open terminal and go to project folder
- Go to Base_Model folder in the project i.e. ```cd Base_Model```
- Run `python text_cnn.py` command
- Go to App folder in the project i.e. `cd ../App`
- Run `streamlit run web_app.py` command
- Enter news headline and click on predict 

## Experiment results
- Open terminal and go to project folder
- Go to Experiment/Differnt_GloVe_Dimension folder i.e. ```cd Experiment/Differnt_GloVe_Dimension/```
- Run Text_CNN.ipynb
- Go to App folder in Experiment folder i.e. ```cd ../App```
- Run `streamlit run web_app.py 50D` command
- Run `streamlit run web_app.py 100D` command
- Run `streamlit run web_app.py 200D` command
- Run `streamlit run web_app.py 300D` command
