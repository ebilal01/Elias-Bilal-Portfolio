import numpy as np
import pandas as pd
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse
from flask import jsonify, request
from pydantic import BaseModel
import uvicorn
import os
from starlette.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json
import logging
from fastapi import FastAPI, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter
import logging
from fastapi import Body
import csv
import aiohttp
from fastapi import FastAPI, Response

router = APIRouter()

class LungCancerData(BaseModel):
    gender: int
    alcohol_use: int
    smoking_level: int
    city: str
# Define your data model
class CityData(BaseModel):
    cities: list[str]


app = FastAPI()

# Setup templates directory
templates = Jinja2Templates(directory="templates")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


# Setup templates directory
templates = Jinja2Templates(directory="frontend")
# Enable CORS

origins = [
    "http://127.0.0.1:5502",  # Add your frontend origin here
    "http://127.0.0.1:5503",
    "http://127.0.0.1:5503/lungcancer.html",
     # Add your Render frontend origin here
    "http://elias-bilal-portfolio-2.onrender.com",
    "https://elias-bilal-portfolio-2.onrender.com"   # Add the HTTP version if needed
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Load CSV files
filename = 'static/data/cancer patient data sets.csv'
patient_df = pd.read_csv(filename, encoding="ISO-8859-1", index_col=0)

dilename = 'static/data/cities_air_quality_water_pollution.18-10-2021.csv'
unit_df = pd.read_csv(dilename, encoding="ISO-8859-1", index_col=0)

# creates new separate data frames for the air quality per each city in the csv file
airQuality_citydf = unit_df["AirQuality"]
city = airQuality_citydf.index.values

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)

# sets the probability levels Low, Medium, and High as the numbers 1, 2, and 3
mapping = {'Low': 1, 'Medium': 2, 'High': 3}
patient_df = patient_df.replace({'Level': mapping})

# creates new separate data frames for the level of smoking, probability level, alcohol use, and air pollution of the patient data
probabilityLevel = patient_df["Level"]
smokinglevel = patient_df["Smoking"]
dranklevel = patient_df["Alcohol use"]
pollute_level = patient_df["Air Pollution"]

# creates new separate data frames for the level of smoking, alcohol use, and air pollution of the patient data but also merges each new dataframe with the probability level (1-3) of getting lung cancer for each patient
combine = pd.merge(probabilityLevel, smokinglevel, right_index=True, left_index=True)
dombine = pd.merge(probabilityLevel, dranklevel, right_index=True, left_index=True)
tombine = pd.merge(probabilityLevel, pollute_level, right_index=True, left_index=True)

# creates new empty dataframes to be filled with the probability level of getting lung cancer for each level of either smoking, drinking, or air pollution exposure of patients
smoke_point_df = pd.DataFrame(
    {
        "SmokingLevel": [],
    },
    columns=["SmokingLevel"]
)
drank_point_df = pd.DataFrame(
    {
        "DrinkingLevel": [],
    },
    columns=["DrinkingLevel"]
)
qual_point_df = pd.DataFrame(
    {
        "QualityLevel": [],
    },
    columns=["QualityLevel"]
)

# fills the empty data frames with the average lung cancer probability level of patients with a certain smoking level
for x in range(1, 9):
    popsmoke_df = combine.loc[combine['Smoking'] == x]
    smokelevel_df = popsmoke_df["Level"]
    totsmoke = smokelevel_df.sum()
    row10 = len(smokelevel_df.index)
    average_level = totsmoke / row10
    smoke_point_df.loc[x] = average_level
    print(smoke_point_df)

# fills the empty data frames with the average lung cancer probability level of patients with a certain drinking level
for x in range(1, 9):
    drank_df = dombine.loc[dombine['Alcohol use'] == x]
    dranklevel_df = drank_df["Level"]
    totsdrank = dranklevel_df.sum()
    row11 = len(dranklevel_df.index)
    average_devel = totsdrank / row11
    drank_point_df.loc[x] = average_devel
    print(drank_point_df)

# fills the empty data frames with the average lung cancer probability level of patients with a certain air pollution exposure level
for x in range(1, 7):
    qual_df = tombine.loc[tombine['Air Pollution'] == x]
    quallevel_df = qual_df["Level"]
    totsqual = quallevel_df.sum()
    row12 = len(quallevel_df.index)
    average_bevel = totsqual / row12
    qual_point_df.loc[x] = average_bevel
    print(qual_point_df)

# defines the probabilities of getting lung cancer based on smoking level as a variable and the index of how many smoking levels there are as variables
cancerprob = smoke_point_df["SmokingLevel"]
index_smoke = smoke_point_df.index.values

# defines the probabilities of getting lung cancer based on drinking level as a variable and the index of how many drinking levels there are as variables
dancerprob = drank_point_df["DrinkingLevel"]
index_drank = drank_point_df.index.values

# defines the probabilities of getting lung cancer based on pollution exposure level as a variable and the index of how many pollution levels there are as variables
tancerprob = qual_point_df["QualityLevel"]
index_qual = qual_point_df.index.values

# graphs the lung cancer probabilities per level of smoking
y = cancerprob
x = index_smoke
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, y, 'ro', label='probability level')
plt.plot(x, p(x))
plt.xlabel('smoking level (1-8) 8 is the highest')
plt.ylabel('Risk Level of Getting Lung Cancer (1-3) with 3 being highest')
plt.title("Lung Cancer Probability verses Smoking", fontsize=15)
plt.savefig("static/graphs/smoking_prob.png")
plt.close()

# graphs the lung cancer probabilities per level of drinking
y = dancerprob
x = index_drank
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, y, 'ro', label='probability level')
plt.plot(x, p(x))
plt.xlabel('drinking level (1-6) with 6 being the highest')
plt.ylabel('Risk Level of Getting Lung Cancer (1-3) with 3 being highest')
plt.title("Lung Cancer Probability verses Drinking", fontsize=15)
plt.savefig("static/graphs/drinking_prob.png")
plt.close()

# graphs the lung cancer probabilities per level of air pollution exposure
y = tancerprob
x = index_qual
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, y, 'ro', label='probability level')
plt.plot(x, p(x))
plt.xlabel('Pollution level 1-6 with 6 being the highest')
plt.ylabel('Risk Level of Getting Lung Cancer (1-3) with 3 being highest')
plt.title("Lung Cancer Probability verses Air Pollution", fontsize=15)
plt.savefig("static/graphs/Air_Qual_prob.png")
plt.close()

# Define your API endpoints
# Define API endpoints
# Endpoint for returning HTML form
# Define your API endpoints
# Endpoint for returning HTML form
# Define your endpoint to get cities
@app.get("/cities", response_model=CityData)
async def get_cities():
    # Fetch cities from your data source (e.g., database)
    cities = unit_df["City"] # Replace this with your actual list of cities
    return {"cities": cities}

@app.get("/favicon.ico")
async def get_favicon():
    return Response(content=b"", media_type="image/x-icon")

@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    
    with open("lungcancer.html", "r") as file:
        return file.read()
@app.post("/submit")
async def submit_form(data: LungCancerData = Body(...)):
    try:
        gender = data.gender
        alcohol_use = data.alcohol_use
        smoking_level = data.smoking_level
        city = data.city
        # Now you can use these variables as needed within the function
        print("Gender:", gender)
        print("Alcohol use:", alcohol_use)
        print("Smoking level:", smoking_level)
        print("City:", city)
        # You can directly access the parsed form data from the 'data' parameter
        first_df = pd.DataFrame(
            {
                "Alcohol Use": [alcohol_use],
                "Smoking": [smoking_level],
                "Gender": [gender],
                "City": [city]
            },
            columns=["Alcohol Use", "Smoking", "Gender", "City"]
        )

        # sets the city air quality of where they live equal to a variable and prints it
        pollution = airQuality_citydf.loc[city]
        print("The air quality level of your city is", pollution)

        pol_level = round((pollution* 6)/100)
        if pol_level == 1:
            airquality = 6
        if pol_level == 2:
            airquality = 5
        if pol_level == 3:
            airquality = 4
        if pol_level == 4:
            airquality = 3
        if pol_level == 5:
            airquality = 2
        if pol_level == 6:
            airquality = 1
        # prints the user's information dataframe
        print(first_df)  
        filter1_df = patient_df.loc[patient_df['Smoking'] == smoking_level]
        print("Filter 1",filter1_df["Level"])    
        filter2_df = filter1_df.loc[filter1_df['Gender'] == gender]
        print("Filter 2",filter2_df["Patient Id"])
        filter3_df = filter2_df.loc[filter2_df['Alcohol use'] == alcohol_use]
        print("Filter 3",filter3_df["Patient Id"])
        filter4_df = filter3_df.loc[filter3_df['Air Pollution'] == airquality]
        print("Filter 4 patient ids = ",filter4_df["Patient Id"])
        print ("Filter 4 levels = ",filter4_df["Level"])

        print("Filter 3 (alchol use filter ) patient ids = ",filter3_df["Patient Id"])
        print ("Filter 3 (alchol use filter) levels = ",filter3_df["Level"])
        # Isolates the levels of each patient from the filtered dataframe
        level_df = filter4_df["Level"]
        level3_df = filter3_df["Level"]
        level2_df = filter2_df["Level"]
        # sets the number of rows in the filtered dataframe equal to row(x) and the number of collumns equal to collumn(x)
        row, col = filter4_df.shape
        row3, col3 = filter3_df.shape
        row2, col2 = filter2_df.shape
        print("number of matched patients for Air Quality Use  filter = " + str(row))
        print("number of matched patients for Alcohol Use  filter = " + str(row3 ))
        print("number of matched patients for Sex filter = " + str(row2))
        # takes an average of the probability levels of each matched patient 
        Total = level_df.sum()
        Total3 = level3_df.sum()
        Total2 = level2_df.sum()

        print("total of all levels in Airquality filter =" + str(Total))
        print("total of all levels in Alcohol use filter =" + str(Total3))
        print("total of all levels in Sex filter =" + str(Total2))
        # if row3 is = to zero there are no matched patients in the filter3 dataframe, so the matched patients from the previous filter have their probability levels averaged then depending on the magnitude of that average from the matched patients the user is told their own probability of getting cancer 
    
        if row == 0 and row3 == 0:
            ave = (Total2/row2)
            print ("filter 2 average = " + str(ave))
            print ("filter2 levels = " + str(filter2_df["Level"]))
            preanswer = "No similar patient data available for your alcohol use and the airquality of your city but based on the factors of Smoking Level, Alcohol Use: "
            if ave < 1.5:
                preanswer = "No similar patient data available for your alcohol use and the airquality of your city but based on the factors of Smoking Level and Sex: "
                answer = preanswer + "You have a Low probability of developing Lung Cancer"
                result = JSONResponse(content={"message": answer})
                return result
            if ave >= 1.5 and ave < 2.5:
                preanswer = "No similar patient data available for your alcohol use and the airquality of your city but based on the factors of Smoking Level and Sex: "
                answer = preanswer + "You have a Medium probability of developing Lung Cancer"
                result = JSONResponse(content={"message": answer})
                return result
            if ave >= 2.5:
                preanswer = "No similar patient data available for your alcohol use and the airquality of your city but based on the factors of Smoking Level and Sex: "
                answer = preanswer + "You have a High probability of developing Lung Cancer" 
                result = JSONResponse(content={"message": answer})
                return result

        # if row is = to zero there are no matched patients in the filter4 dataframe, so the matched patients from the previous filter have their probability levels averaged then depending on the magnitude of that average from the matched patients the user is told their own probability of getting cancer 
        if row == 0:
            print(Total3)
            Average = (Total3/row3)
            print (filter3_df["Level"])
            preanswer = "No similar patient data available for the airquality of your city but based on the factors of Smoking Level, Alcohol Use, and Sex: "
            if Average < 1.5:
                preanswer = "No similar patient data available for the airquality of your city but based on the factors of Smoking Level, Alcohol Use, and Sex: "
                answer = preanswer + "You have a Low probability of developing Lung Cancer"
                result = JSONResponse(content={"message": answer})
                return result
            if Average >= 1.5 and Average < 2.5:
                preanswer = "No similar patient data available for the airquality of your city but based on the factors of Smoking Level, Alcohol Use, and Sex: "
                answer = preanswer + "You have a Medium probability of developing Lung Cancer"
                result = JSONResponse(content={"message": answer})
                return result
            if Average >= 2.5:
                preanswer = "No similar patient data available for the airquality of your city but based on the factors of Smoking Level, Alcohol Use, and Sex: "
                answer = preanswer + "You have a High probability of developing Lung Cancer"
                result = JSONResponse(content={"message": answer})
                return result

        # if row = 1 there is one matched patient in the filter4 dataframe, so there is no need to take an average then depending on the magnitude of that level from the matched patient the user is told their own probability of getting cancer 
        elif row == 1:
            print(filter4_df["Level"])
            level_value = filter4_df["Level"].iloc[0]
            if level_value < 1.5:
                answer = "You have a Low probability of developing Lung Cancer"
                result = JSONResponse(content={"message": answer})
                return result
            if level_value >= 1.5 and level_value < 2.5:
                answer = "You have a Medium probability of developing Lung Cancer"
                result = JSONResponse(content={"message": answer})
                return result
            if level_value >= 2.5:
                answer = "You have a High probability of developing Lung Cancer"
                result = JSONResponse(content={"message": answer})
                return result

        # if row > 1 there are multiple matched patients in the filter4 dataframe, so the matched patients from filter4 have their probability levels averaged then depending on the magnitude of that average from the matched patients the user is told their own probability of getting cancer 
        elif row > 1:
            average = (Total/row)
            if average < 1.5:
                answer ="You have a Low probability of developing Lung Cancer"
                result = JSONResponse(content={"message": answer})
                return result
            if average >= 1.5 and average < 2.5:
                answer = "You have a Medium probability of developing Lung Cancer"
                result = JSONResponse(content={"message": answer})
                return result
            if average >= 2.5:
                answer ="You have a High probability of developing Lung Cancer"
                result = JSONResponse(content={"message": answer})
                return result

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# Run the FastAPI app
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=os.getenv("PORT", 5502))