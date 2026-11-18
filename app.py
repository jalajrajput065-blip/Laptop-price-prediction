import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    with (BASE_DIR / "pipe.pkl").open("rb") as model_file:
        return pickle.load(model_file)


@st.cache_data
def load_options():
    df = pd.read_csv(BASE_DIR / "laptop_data.csv")

    cpu_names = df["Cpu"].apply(lambda text: " ".join(text.split()[0:3]))

    def fetch_processor(text):
        if text in ("Intel Core i7", "Intel Core i5", "Intel Core i3"):
            return text
        return "Other Intel Processor" if text.split()[0] == "Intel" else "AMD Processor"

    def cat_os(text):
        if text in ("Windows 10", "Windows 7", "Windows 10 S"):
            return "Windows"
        if text in ("macOS", "Mac OS X"):
            return "Mac"
        return "Others/No OS/Linux"

    return {
        "Company": sorted(df["Company"].unique()),
        "TypeName": sorted(df["TypeName"].unique()),
        "Cpu brand": sorted(cpu_names.apply(fetch_processor).unique()),
        "Gpu brand": sorted(df["Gpu"].apply(lambda text: text.split()[0]).unique()),
        "os": sorted(df["OpSys"].apply(cat_os).unique()),
    }


try:
    pipe = load_model()
    options = load_options()
except Exception as exc:
    st.error(f"Could not load the prediction model: {exc}")
    st.stop()

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand', options["Company"])

# type of laptop
type = st.selectbox('Type', options["TypeName"])

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop', min_value=0.5, max_value=10.0, value=1.5, step=0.1)

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.slider('Screen size in inches', 10.0, 18.0, 13.0)

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU', options["Cpu brand"])

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU', options["Gpu brand"])

os = st.selectbox('OS', options["os"])

if st.button('Predict Price'):
    # query
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = pd.DataFrame([[
        company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os
    ]], columns=[
        "Company", "TypeName", "Ram", "Weight", "Touchscreen", "Ips",
        "ppi", "Cpu brand", "HDD", "SSD", "Gpu brand", "os"
    ])
    prediction = int(np.exp(pipe.predict(query)[0]))
    st.title(f"The predicted price of this configuration is {prediction}")

