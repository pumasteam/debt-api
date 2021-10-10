from tensorflow.python.framework.func_graph import flatten
from tensorflow.python.ops.gen_array_ops import InvertPermutation
from fastapi import FastAPI
from tensorflow import keras
from ModelStructure import RegressorNN
import numpy as np
import uvicorn

def get_model(n_features):
    model = RegressorNN(n_features)
    model.build(input_shape = (None, n_features))
    return model

# load a ton of models
ny_sat_high = get_model(3)
ny_sat_high.load_weights('models/ny/ny_sat_high.h5')
ny_sat_mid = get_model(3)
ny_sat_mid.load_weights('models/ny/ny_sat_mid.h5')
ny_sat_low = get_model(3)
ny_sat_low.load_weights('models/ny/ny_sat_low.h5')
ny_act_high = get_model(3)
ny_act_high.load_weights('models/ny/ny_act_high.h5')
ny_act_mid = get_model(3)
ny_act_mid.load_weights('models/ny/ny_act_mid.h5')
ny_act_low = get_model(3)
ny_act_low.load_weights('models/ny/ny_act_low.h5')

ca_sat_high = get_model(3)
ca_sat_high.load_weights('models/ca/ca_sat_high.h5')
ca_sat_mid = get_model(3)
ca_sat_mid.load_weights('models/ca/ca_sat_mid.h5')
ca_sat_low = get_model(3)
ca_sat_low.load_weights('models/ca/ca_sat_low.h5')
ca_act_high = get_model(3)
ca_act_high.load_weights('models/ca/ca_act_high.h5')
ca_act_mid = get_model(3)
ca_act_mid.load_weights('models/ca/ca_act_mid.h5')
ca_act_low = get_model(3)
ca_act_low.load_weights('models/ca/ca_act_low.h5')

ma_sat_high = get_model(3)
ma_sat_high.load_weights('models/ma/ma_sat_high.h5')
ma_sat_mid = get_model(3)
ma_sat_mid.load_weights('models/ma/ma_sat_mid.h5')
ma_sat_low = get_model(3)
ma_sat_low.load_weights('models/ma/ma_sat_low.h5')
ma_act_high = get_model(3)
ma_act_high.load_weights('models/ma/ma_act_high.h5')
ma_act_mid = get_model(3)
ma_act_mid.load_weights('models/ma/ma_act_mid.h5')
ma_act_low = get_model(3)
ma_act_low.load_weights('models/ma/ma_act_low.h5')

il_sat_high = get_model(3)
il_sat_high.load_weights('models/il/il_sat_high.h5')
il_sat_mid = get_model(3)
il_sat_mid.load_weights('models/il/il_sat_mid.h5')
il_sat_low = get_model(3)
il_sat_low.load_weights('models/il/il_sat_low.h5')
il_act_high = get_model(3)
il_act_high.load_weights('models/il/il_act_high.h5')
il_act_mid = get_model(3)
il_act_mid.load_weights('models/il/il_act_mid.h5')
il_act_low = get_model(3)
il_act_low.load_weights('models/il/il_act_low.h5')
# model dictionary
model_dict = {
    'sat':{
        'california':{
            'low':ca_sat_low,
            'mid':ca_sat_mid,
            'high':ca_sat_high
        },
        'illinois':{
            'low':il_sat_low,
            'mid':il_sat_mid,
            'high':il_sat_high
        },
        'massachusetts':{
            'low':ma_sat_low,
            'mid':ma_sat_mid,
            'high':ma_sat_high
        },
        'newyork':{
            'low':ny_sat_low,
            'mid':ny_sat_mid,
            'high':ny_sat_high
        }
    },
    'act':{
        'california':{
            'low':ca_act_low,
            'mid':ca_act_mid,
            'high':ca_act_high
        },
        'illinois':{
            'low':il_act_low,
            'mid':il_act_mid,
            'high':il_act_high
        },
        'massachusetts':{
            'low':ma_act_low,
            'mid':ma_act_mid,
            'high':ma_act_high
        },
        'newyork':{
            'low':ny_act_low,
            'mid':ny_act_mid,
            'high':ny_act_high
        }
    }
}
def flatten_prediction(values, model):
    prediction = model.predict(np.array([values]))
    prediction = prediction.reshape(-1)
    return prediction[0]

DEBT_RATE = 1.0373
RATE_6 = DEBT_RATE
RATE_12 = DEBT_RATE*2
RATE_18 = DEBT_RATE*4
RATE_24 = DEBT_RATE*6

app = FastAPI()
@app.get("/total_debt_sat/")
async def get_debt(region: str = 'newyork', 
                annual_income: int = 20000, 
                sat_reading: int = 200, 
                sat_writing: int = 200, 
                sat_math: int = 400):
    
    region = region.lower()
    low_income_dif = 4000
    mid_income_dif = 2000
    if annual_income < 30001:
        model_to_use = model_dict['sat'][region]['low']
    elif annual_income > 30000 and annual_income < 75001:
        model_to_use = model_dict['sat'][region]['mid']
    elif annual_income > 75000:
        model_to_use = model_dict['sat'][region]['high']
    prediction = flatten_prediction([sat_reading*2, sat_math, sat_writing*2], model_to_use)
    if annual_income < 30001:
        prediction+=low_income_dif
    elif annual_income > 30000 and annual_income < 75001:
        prediction+=mid_income_dif
    return {"total_debt":int(prediction)}

@app.get("/monthly_debt_sat/")
async def monthly_debt(region: str = 'newyork', 
                annual_income: int = 20000, 
                sat_reading: int = 200, 
                sat_writing: int = 200, 
                sat_math: int = 400):
    #### SAME CODE AS TOTAL DEBT ####
    region = region.lower()
    low_income_dif = 4000
    mid_income_dif = 2000
    if annual_income < 30001:
        model_to_use = model_dict['sat'][region]['low']
    elif annual_income > 30000 and annual_income < 75001:
        model_to_use = model_dict['sat'][region]['mid']
    elif annual_income > 75000:
        model_to_use = model_dict['sat'][region]['high']
    prediction = flatten_prediction([sat_reading*2, sat_math, sat_writing*2], model_to_use)
    if annual_income < 30001:
        prediction+=low_income_dif
    elif annual_income > 30000 and annual_income < 75001:
        prediction+=mid_income_dif
    ##################################
    prediction_int = int(prediction)
    # monthly debt interest
    monthly = int(prediction_int/24) # if you pay in 24 months
    month_6 = int(monthly*RATE_6)
    month_12 = int(monthly*RATE_12)
    month_18 = int(monthly*RATE_18)
    month_24 = int(monthly*RATE_24)

    return {"monthly_debt": [month_6, month_12, month_18, month_24]}

# print(flatten_prediction([10, 10, 20], ny_act_low))
if __name__ == "__main__":
    uvicorn.run(app, port = 5000,host='0.0.0.0')