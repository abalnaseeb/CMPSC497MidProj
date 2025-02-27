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
