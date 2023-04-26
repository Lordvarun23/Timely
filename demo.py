import os

import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from csv import DictReader
from FeatureEngineeing import FE
import csv
from Models import Model

app = FastAPI()

@app.get("/")
async def read_form():
    html = """
    <html>
    <head>
    <title> Time Series Model </title>
    <body bgcolor = #A1A5F1 >
        <center> <h1><marquee> <u>DataGenie Hackathon </u></marquee></h1></center>
        
        <h2> Upload CSV File </h2>
        <h3> Points to be Remembered: </h3>
        <ol>
        <li>The Data File should be in csv format </li>
        <li>The Data File Should have only 2 columns </li>
        <li>The Data File Columns name should only be [point_timestamp,point_value]</li>
        <li> The Data File Date Format should be YYYY-MM-DD </li>
        </ol> 
        <form action="/uploadfile/" enctype="multipart/form-data" method="post">
        <input name="file" type="file">
        <p>typeData: 0-> Day 1->Month 2->Weakly 3-> Hourly</p>
        <input name="type" type="text" placeholder="Enter your file type" >
        <h2>Date to be Predicted: </h2>
        <input name="date" type="text" placeholder="Enter Date to be forecasted" >
        <input type="submit">
        </form>
    </body>
    </head>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)

app.mount("/static", StaticFiles(directory="C:\\Users\\Public\\pythonProject\\FastapiDemo\\uploadfile\\static"), name="static")
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...),type: str = Form(...),date: str = Form(...)):
    contents = await file.read()
    decoded = contents.decode('utf-8').splitlines()
    reader = csv.reader(decoded)
    data = [row for row in reader]
    for i in range(len(data)):
        data[i] = data[i][1:]
    cols = data[0]
    data = data[1:]
    df = pd.DataFrame(data,columns=cols)
    df.to_csv("result.csv",index=False)
    print("Success")
    best_model = FE(type)
    df.set_index("point_timestamp", inplace=True)
    df.index = pd.to_datetime(df.index)
    df.dropna(inplace=True)
    mape_best,forecaste  = Model(best_model,df,date)


    html = f'''
        <html>
        <head>
        <title> Time Series Model </title>
        <body bgcolor = #A1A5F1 >
        <center> <h1><marquee> <u>DataGenie Hackathon </u></marquee></h1></center>
        <center> <h2><u>Results </u></h2></center>
        <h2><u>Results: </u></h2>
        <ol>
        <li> Best Model: {best_model} </li>
        <li> Type of File: {type} </li>
        <li> MAPE Score of Test by Splitting in ratio 90:10 : {mape_best} </li>
        <li> Prediction for the date {date} is approximately equal to {forecaste}</li>
        </ol>
        </body>
        </head>
        </html>
        '''
    return HTMLResponse(content = html,status_code=200)
