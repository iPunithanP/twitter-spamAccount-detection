# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:21:47 2023

@author: punit
"""

import numpy as np
import pickle
import gender_guesser.detector as gender

def predict_sex(name):
    name = str(name)
    sex_predictor = gender.Detector(case_sensitive=False)
    first_name = name.split(' ')[0]
    sex = sex_predictor.get_gender(first_name)
    sex_dict = {'female': -2, 'mostly_female': -1, 'unknown': 0, 'mostly_male': 1, 'male': 2}
    sex_code = sex_dict[sex]
    return sex_code


loaded_model = pickle.load(open('C:/Users/punit/OneDrive/Desktop/project/bestmodel.sav', 'rb'))

name = "Noob gamer"
sex_code = predict_sex(name)
print(sex_code)

input_data = (24, 4, 588, 16, 0, sex_code, 1)
np_arr = np.asarray(input_data)
arr_reshape = np_arr.reshape(1,-1)

prediction = loaded_model(arr_reshape)

print(prediction)

if (prediction[0] == 0):
    print('The account is legit')
else:
    print('The account is fake')