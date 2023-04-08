# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:31:32 2023

@author: punit
"""

import numpy as np
import pickle
import streamlit as st
import gender_guesser.detector as gender


loaded_model = pickle.load(open('C:/Users/punit/OneDrive/Desktop/project/bestmodel.sav', 'rb'))
#creating a fn for prediction

def predict_sex(name):
    name = str(name)
    sex_predictor = gender.Detector(case_sensitive=False)
    first_name = name.split(' ')[0]
    sex = sex_predictor.get_gender(first_name)
    sex_dict = {'female': -2, 'mostly_female': -1, 'unknown': 0, 'mostly_male': 1, 'male': 2}
    sex_code = sex_dict[sex]
    return sex_code

def bot_prediction(input_data):
    
    #input_data = (24, 4, 588, 16, 0, 0, 1)
    np_arr = np.asarray(input_data)
    arr_reshape = np_arr.reshape(1,-1)

    prediction = loaded_model(arr_reshape)

    print(prediction)

    if (prediction[0] == 0):
        return 'The account is legit'
    else:
        return 'The account is fake'
    

def main():
    
    # Title for the webpage
    st.title('Twitter Fake Account Prediction')
    
    #getting inputs
    # statuses_count, followers_count, friends_count, favourites_count, listed_count, sex_code, lang_code
    name = st.text_input('Profile name')
    Sex_code = predict_sex(name)
    Statuses_count = st.text_input('Number of Tweets')
    Followers_count = st.text_input('Number of Followers')
    Friends_count = st.text_input('Number of Followings')
    Favourites_count = st.text_input('Number of Favourites')
    Listed_count = st.text_input('Number of Comments')
    Lang_code = st.text_input('Language code: English: 1, French: 2, 	Italian: 3, Spanish: 4, others: 0' )
    
    #code for prediction
    detection = ''
    
    #creating a button for prediction
    
    if st.button('Account Verification Result'):
        detection = bot_prediction([Statuses_count, Followers_count, Friends_count, Favourites_count, Listed_count, Sex_code, Lang_code])
        
    st.success(detection)
    
if __name__ == '__main__':
    main()