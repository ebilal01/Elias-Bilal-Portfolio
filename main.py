import numpy as np
import pandas as pd
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse
from flask import jsonify
import uvicorn
import os
from starlette.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json
import logging
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi import Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

class LungCancerData(BaseModel):
    gender: int
    alcohol_use: int
    smoking_level: int
    city: str

app = FastAPI()

# Setup templates directory
templates = Jinja2Templates(directory="templates")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware
origins = [
    "http://127.0.0.1:5500",  # Your local development environment
    "https://ebilal01.github.io",  # Your GitHub Pages site
    # Add any other domains that need to access your API
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
@app.get("/lungcancer-form", response_class=HTMLResponse)
async def get_lungcancer_form():
    return RedirectResponse("/lungcancer.html")

@app.post("/lungcancer")
async def lung_cancer_prediction(data: LungCancerData, request: Request):
    gender = data.gender
    alcohol_use = data.alcohol_use
    smoking_level = data.smoking_level
    city = data.city
    try:
        logging.debug("Received POST request")
        # Retrieve values from the form
        form_data = await request.form()
        
        gender = int(form_data['gender'])
        alcohol_use = int(form_data['alcohol_use'])
        smoking_level = int(form_data['smoking_level'])
        city = form_data['city']
        # Rest of your code here using the retrieved values
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

        # changes the air quality level from the city data set into a level 1-6 of air pollution from the patient data set
        pol_level = round((pollution * 6) / 100)
        airquality = 7 - pol_level

        print(airquality)
        # prints the user's information dataframe
        print(first_df)


        # filters the overall patient data and selects patients who have the same smoking, drinking, sex, and air pollution exposure and puts those patients' data into a new dataframe
        filter1_df = patient_df.loc[patient_df['Smoking'] == smoking_level]
        print("Filter 1", filter1_df["Level"])

        filter2_df = filter1_df.loc[filter1_df['Gender'] == gender]
        print("Filter 2", filter2_df["Patient Id"])

        filter3_df = filter2_df.loc[filter2_df['Alcohol use'] == alcohol_use]
        print("Filter 3", filter3_df["Patient Id"])

# Now perform the comparison
        filter4_df = filter3_df.loc[filter3_df['Air Pollution'] == airquality]
        print("Filter 4", filter4_df["Patient Id"])
        print("Filter 4", filter4_df["Level"])

        # Isolates the levels of each patient from the filtered dataframe
        level_df = filter4_df["Level"]

        # sets the number of rows in the filtered dataframe equal to row(x) and the number of columns equal to column(x)
        row, _ = filter4_df.shape

        # takes an average of the probability levels of each matched patient
        Total = level_df.sum()

        # if row is equal to zero...
        if row == 0:
            print("No similar patient data available")
            answer = "No similar patient data available"
            return JSONResponse(content={"message": answer})
        # if row > 0...
        else:
            average = Total / row
            if average < 1.5:
                answer = "You have a Low probability of developing Lung Cancer"
            elif average < 2.5:
                answer = "You have a Medium probability of developing Lung Cancer"
            else:
                answer = "You have a High probability of developing Lung Cancer"
            return JSONResponse(content={"message": answer})
        
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)