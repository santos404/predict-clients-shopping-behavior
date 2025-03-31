#install venv before running code on editor to  Create a Virtual Environment:
python -m venv venv
#Activate the Virtual Environment:
.\venv\Scripts\activate
#Verify Activation:You should see (venv) at the beginning of your terminal prompt, indicating the environment is active.
#Install Required Libraries:Once activated, install the necessary packages:
pip install streamlit scikit-learn joblib pandas numpy
#Run the Streamlit app:
streamlit run app.py
