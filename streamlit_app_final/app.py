import random
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import os
import numpy as np
import datetime
from scipy.stats import skewnorm

# Setup Streamlit Pages
st.set_page_config(page_title="Theme Park Simulation Dashboard", layout="wide")

# Ensure session state for shared simulation variables
if "selected_day" not in st.session_state:
    st.session_state.selected_day = None
if "switcher_percentage" not in st.session_state:
    st.session_state.switcher_percentage = 0.2

# -------------------------------
# DATA LOADING FUNCTIONS
# -------------------------------
@st.cache_data
def load_simulation_data():
    return pd.read_csv("people_in_line.csv")

# Load simulation data (for simulation pages)
df = load_simulation_data()

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", [
    "Home",
    "Predicting Park Attendance",
    "Predicting Waiting Times",
    "Simulation: Overview Park", 
    "Simulation: Dynamics Averages", 
    "Simulation: View Attraction"
    
])

# -------------------------------
# GLOBAL SETTINGS FOR SIMULATION PAGES
# -------------------------------
if page in ["Simulation: Overview Park", "Simulation: Dynamics Averages", "Simulation: View Attraction"]:
    st.sidebar.header("Global Settings")
    st.session_state.selected_day = st.sidebar.selectbox(
        "Select a Date:", 
        df["DATE"].unique(), 
        index=0 if st.session_state.selected_day is None 
                else list(df["DATE"].unique()).index(st.session_state.selected_day)
    )
    st.session_state.switcher_percentage = st.sidebar.slider(
        "Select % of people using Gamification system", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.switcher_percentage, 
        step=0.05
    )
    # Filter Data for Selected Date and sort time slots
    day_df = df[df["DATE"] == st.session_state.selected_day]
    day_df['TIME SLOT'] = pd.to_datetime(day_df['START OF 1 RIDE FOR ATTRACTION']).dt.time
    day_df = day_df.sort_values(by=['START OF 1 RIDE FOR ATTRACTION'])
    unique_time_slots = sorted(day_df['TIME SLOT'].unique())
    filtered_time_slots = [t for t in unique_time_slots if t != pd.to_datetime('09:00:00').time()]

# -------------------------------
# PAGE: Simulation: Overview Park
# -------------------------------
if page == "Simulation: Overview Park":
    st.title("Theme Park Simulation: Overview Park")
    selected_time_slot = st.selectbox("Select Time Slot:", filtered_time_slots)

    # Global attendance value
    attendance = 20000

    def initialize_attractions(df, start_time='09:00:00', attendance=10000):
        start_time = pd.to_datetime(start_time).time()
        initial_df = df[df['TIME SLOT'] == start_time][['TIME SLOT', 'ATTRACTION', 'WAIT TIME', 'PEOPLE IN LINE', 'PEOPLE INSIDE ATTRACTION']].copy()
        people_in_line = initial_df['PEOPLE IN LINE'].sum()
        people_inside = initial_df['PEOPLE INSIDE ATTRACTION'].sum()
        outside_count = attendance - (people_in_line + people_inside)
        outside_row = pd.DataFrame({
            'TIME SLOT': [start_time],
            'ATTRACTION': ['OUTSIDE'], 
            'WAIT TIME': [0], 
            'PEOPLE IN LINE': [0], 
            'PEOPLE INSIDE ATTRACTION': [outside_count]
        })
        initial_df = pd.concat([initial_df, outside_row], ignore_index=True)
        initial_df = initial_df.sort_values(by='ATTRACTION').reset_index(drop=True)
        return initial_df

    initial_state_09_00 = initialize_attractions(day_df, attendance)

    def process_selected_time_slot(df, selected_time_slot, attendance):
        current_time = selected_time_slot
        next_time_index = unique_time_slots.index(current_time)
        next_time = unique_time_slots[next_time_index] if next_time_index is not None else None
        if next_time:
            current_df = df[df['TIME SLOT'] == current_time].copy()
            next_df = df[df['TIME SLOT'] == next_time][['ATTRACTION', 'WAIT TIME', 'PEOPLE IN LINE', 'PEOPLE INSIDE ATTRACTION']].copy()
            next_df = next_df.sort_values(by='ATTRACTION').reset_index(drop=True)
            current_df['DEMAND'] = current_df['WAIT TIME'].apply(lambda x: 'HIGH' if x > 45 else ('LOW' if x < 15 else 'MEDIUM'))
            next_df['DEMAND'] = next_df['WAIT TIME'].apply(lambda x: 'HIGH' if x > 45 else ('LOW' if x < 15 else 'MEDIUM'))
            total_switchers = st.session_state.switcher_percentage * current_df[current_df['DEMAND'] == 'HIGH']['PEOPLE IN LINE'].sum()
            num_low_demand = next_df[next_df['DEMAND'] == 'LOW'].shape[0]
            switchers_per_low_attraction = total_switchers / num_low_demand if num_low_demand > 0 else 0

            next_df['NEW PEOPLE INSIDE ATTRACTION'] = next_df.apply(
                lambda row: row['PEOPLE INSIDE ATTRACTION'] + switchers_per_low_attraction if row['DEMAND'] == 'LOW' 
                else row['PEOPLE INSIDE ATTRACTION'], axis=1).round()
            next_df['NEW PEOPLE IN LINE'] = next_df.apply(
                lambda row: row['PEOPLE IN LINE'] - (row['PEOPLE IN LINE'] * st.session_state.switcher_percentage) if row['DEMAND'] == 'HIGH' 
                else row['PEOPLE IN LINE'], axis=1).round()
            next_df['NEW WAIT TIME'] = next_df.apply(
                lambda row: 15 * (row['NEW PEOPLE IN LINE'] / (row['NEW PEOPLE INSIDE ATTRACTION'] + 1)) if row['DEMAND'] == 'HIGH' 
                else row['WAIT TIME'], axis=1).round()

            st.subheader(f"State for {next_time}")
            st.dataframe(next_df[['ATTRACTION', 'DEMAND', 'PEOPLE IN LINE', 'PEOPLE INSIDE ATTRACTION', 
                                    'WAIT TIME', 'NEW PEOPLE IN LINE', 'NEW PEOPLE INSIDE ATTRACTION', 'NEW WAIT TIME']])

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            wait_time_max = max(next_df['WAIT TIME'].max(), next_df['NEW WAIT TIME'].max())
            axes[0, 0].bar(next_df['ATTRACTION'], next_df['WAIT TIME'], color=next_df['DEMAND'].apply(lambda x: 'red' if x == 'HIGH' else 'black'))
            axes[0, 0].set_title(f'Wait Time at {next_time}')
            axes[0, 0].set_ylim(0, wait_time_max)
            axes[0, 0].tick_params(axis='x', rotation=90)
            axes[0, 1].bar(next_df['ATTRACTION'], next_df['NEW WAIT TIME'], color=next_df['DEMAND'].apply(lambda x: 'orange' if x == 'HIGH' else 'black'))
            axes[0, 1].set_title(f'New Wait Time at {next_time}')
            axes[0, 1].set_ylim(0, wait_time_max)
            axes[0, 1].tick_params(axis='x', rotation=90)
            axes[1, 0].bar(next_df['ATTRACTION'], next_df['PEOPLE INSIDE ATTRACTION'], color=next_df['DEMAND'].apply(lambda x: 'blue' if x == 'LOW' else 'black'))
            axes[1, 0].set_title(f'People Inside Attraction at {next_time}')
            axes[1, 0].tick_params(axis='x', rotation=90)
            axes[1, 1].bar(next_df['ATTRACTION'], next_df['NEW PEOPLE INSIDE ATTRACTION'], color=next_df['DEMAND'].apply(lambda x: 'green' if x == 'LOW' else 'black'))
            axes[1, 1].set_title(f'New People Inside Attraction at {next_time}')
            axes[1, 1].tick_params(axis='x', rotation=90)
            plt.tight_layout()
            st.pyplot(fig)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h4 style='text-align: center;'>Current State</h4>", unsafe_allow_html=True)
                st.write(f"**People inside Attractions:** {next_df['PEOPLE INSIDE ATTRACTION'].sum()}")
                st.write(f"**People in Lines:** {next_df['PEOPLE IN LINE'].sum()}")
                st.write(f"**People Outside:** {attendance - next_df['PEOPLE INSIDE ATTRACTION'].sum() - next_df['PEOPLE IN LINE'].sum()}")
            with col2:
                st.markdown("<h4 style='text-align: center;'>New State After Redistribution</h4>", unsafe_allow_html=True)
                st.write(f"**New People inside Attractions:** {next_df['NEW PEOPLE INSIDE ATTRACTION'].sum()}")
                st.write(f"**New People in Lines:** {next_df['NEW PEOPLE IN LINE'].sum()}")
                st.write(f"**People Outside:** {attendance - next_df['NEW PEOPLE INSIDE ATTRACTION'].sum() - next_df['NEW PEOPLE IN LINE'].sum()}")

    process_selected_time_slot(day_df, selected_time_slot, attendance)

# -------------------------------
# PAGE: Simulation: Dynamics Averages
# -------------------------------
elif page == "Simulation: Dynamics Averages":
    st.title("Theme Park Simulation: Dynamics Average")
    time_slots = sorted(day_df["TIME SLOT"].unique())
    time_labels = []
    mean_waits_original = []
    mean_waits_new = []
    mean_line_original = []
    mean_line_new = []
    mean_inside_original = []
    mean_inside_new = []
    
    def classify_demand(wait):
        if wait > 45:
            return "HIGH"
        elif wait < 15:
            return "LOW"
        else:
            return "MEDIUM"
    
    for i, ts in enumerate(time_slots):
        ts_label = ts.strftime("%H:%M:%S")
        time_labels.append(ts_label)
        df_current = day_df[day_df["TIME SLOT"] == ts].copy()
        df_current["DEMAND"] = df_current["WAIT TIME"].apply(classify_demand)
        if i == 0:
            df_current["NEW_WAIT_TIME"] = df_current["WAIT TIME"]
            df_current["NEW_PEOPLE_IN_LINE"] = df_current["PEOPLE IN LINE"]
            df_current["NEW_PEOPLE_INSIDE"] = df_current["PEOPLE INSIDE ATTRACTION"]
        else:
            prev_ts = time_slots[i - 1]
            df_prev = day_df[day_df["TIME SLOT"] == prev_ts].copy()
            df_prev["DEMAND"] = df_prev["WAIT TIME"].apply(classify_demand)
            total_switchers = st.session_state.switcher_percentage * df_prev.loc[df_prev["DEMAND"] == "HIGH", "PEOPLE IN LINE"].sum()
            num_low_demand = df_current[df_current["DEMAND"] == "LOW"].shape[0]
            switchers_per_low = total_switchers / num_low_demand if num_low_demand > 0 else 0
            new_wait_list = []
            new_line_list = []
            new_inside_list = []
            for idx, row in df_current.iterrows():
                demand = row["DEMAND"]
                if demand == "HIGH":
                    new_line = row["PEOPLE IN LINE"] - (row["PEOPLE IN LINE"] * st.session_state.switcher_percentage)
                    new_wait = 15 * (new_line / (row["PEOPLE INSIDE ATTRACTION"] + 1))
                    new_inside = row["PEOPLE INSIDE ATTRACTION"]
                elif demand == "LOW":
                    new_inside = row["PEOPLE INSIDE ATTRACTION"] + switchers_per_low
                    new_line = row["PEOPLE IN LINE"]
                    new_wait = row["WAIT TIME"]
                else:
                    new_wait = row["WAIT TIME"]
                    new_line = row["PEOPLE IN LINE"]
                    new_inside = row["PEOPLE INSIDE ATTRACTION"]
                new_wait_list.append(new_wait)
                new_line_list.append(new_line)
                new_inside_list.append(new_inside)
            df_current["NEW_WAIT_TIME"] = new_wait_list
            df_current["NEW_PEOPLE_IN_LINE"] = new_line_list
            df_current["NEW_PEOPLE_INSIDE"] = new_inside_list
        
        mean_waits_original.append(df_current["WAIT TIME"].mean())
        mean_waits_new.append(df_current["NEW_WAIT_TIME"].mean())
        mean_line_original.append(df_current["PEOPLE IN LINE"].mean())
        mean_line_new.append(df_current["NEW_PEOPLE_IN_LINE"].mean())
        mean_inside_original.append(df_current["PEOPLE INSIDE ATTRACTION"].mean())
        mean_inside_new.append(df_current["NEW_PEOPLE_INSIDE"].mean())
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(time_labels, mean_waits_original, label="Mean WAIT TIME", marker="o")
    ax1.plot(time_labels, mean_waits_new, label="Mean NEW WAIT TIME", marker="o")
    ax1.set_xlabel("Time Slot")
    ax1.set_ylabel("Wait Time (min)")
    ax1.set_title("Average Wait Time Across Attractions")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(time_labels, mean_line_original, label="Mean PEOPLE IN LINE", marker="o")
    ax2.plot(time_labels, mean_line_new, label="Mean NEW PEOPLE IN LINE", marker="o")
    ax2.set_xlabel("Time Slot")
    ax2.set_ylabel("People Count")
    ax2.set_title("Average People in Line Across Attractions")
    ax2.legend()
    ax2.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(time_labels, mean_inside_original, label="Mean PEOPLE INSIDE", marker="o")
    ax3.plot(time_labels, mean_inside_new, label="Mean NEW PEOPLE INSIDE", marker="o")
    ax3.set_xlabel("Time Slot")
    ax3.set_ylabel("People Count")
    ax3.set_title("Average People Inside Attractions")
    ax3.legend()
    ax3.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig3)

# -------------------------------
# PAGE: Simulation: View Attraction
# -------------------------------
elif page == "Simulation: View Attraction":
    st.title("Theme Park Simulation: View Attraction")
    attractions = sorted(df["ATTRACTION"].unique())
    selected_attraction = st.selectbox("Select Attraction:", attractions)
    time_slots = sorted(day_df["TIME SLOT"].unique())
    
    def classify_demand(wait):
        if wait > 45:
            return "HIGH"
        elif wait < 15:
            return "LOW"
        else:
            return "MEDIUM"
    
    time_labels = []
    original_waits = []
    new_waits = []
    
    for i, ts in enumerate(time_slots):
        time_labels.append(ts.strftime("%H:%M:%S"))
        row = day_df[(day_df["TIME SLOT"] == ts) & (day_df["ATTRACTION"] == selected_attraction)]
        if not row.empty:
            original_wait = row["WAIT TIME"].values[0]
        else:
            original_wait = None
        original_waits.append(original_wait)
        if i == 0:
            new_waits.append(original_wait)
        else:
            prev_ts = time_slots[i - 1]
            prev_df = day_df[day_df["TIME SLOT"] == prev_ts].copy()
            prev_df["DEMAND"] = prev_df["WAIT TIME"].apply(classify_demand)
            curr_df = day_df[day_df["TIME SLOT"] == ts].copy()
            curr_df["DEMAND"] = curr_df["WAIT TIME"].apply(classify_demand)
            total_switchers = st.session_state.switcher_percentage * prev_df[prev_df["DEMAND"] == "HIGH"]["PEOPLE IN LINE"].sum()
            num_low_demand = curr_df[curr_df["DEMAND"] == "LOW"].shape[0]
            switchers_per_low_attraction = total_switchers / num_low_demand if num_low_demand > 0 else 0
            if not row.empty:
                current_wait = row["WAIT TIME"].values[0]
                current_demand = classify_demand(current_wait)
                if current_demand == "HIGH":
                    original_line = row["PEOPLE IN LINE"].values[0]
                    original_inside = row["PEOPLE INSIDE ATTRACTION"].values[0]
                    new_people_in_line = original_line - (original_line * st.session_state.switcher_percentage)
                    new_wait = 15 * (new_people_in_line / (original_inside + 1))
                else:
                    new_wait = current_wait
            else:
                new_wait = None
            new_waits.append(new_wait)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_labels, original_waits, label="WAIT TIME", marker="o")
    ax.plot(time_labels, new_waits, label="NEW WAIT TIME", marker="o")
    ax.set_xlabel("Time Slot")
    ax.set_ylabel("Wait Time (min)")
    ax.set_title(f"Wait Times for {selected_attraction} Over the Day")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    time_labels_inside = []
    original_inside_list = []
    new_inside_list = []
    
    for i, ts in enumerate(time_slots):
        time_labels_inside.append(ts.strftime("%H:%M:%S"))
        row = day_df[(day_df["TIME SLOT"] == ts) & (day_df["ATTRACTION"] == selected_attraction)]
        if not row.empty:
            original_inside = row["PEOPLE INSIDE ATTRACTION"].values[0]
        else:
            original_inside = None
        original_inside_list.append(original_inside)
        if i == 0:
            new_inside_list.append(original_inside)
        else:
            prev_ts = time_slots[i - 1]
            prev_df = day_df[day_df["TIME SLOT"] == prev_ts].copy()
            prev_df["DEMAND"] = prev_df["WAIT TIME"].apply(classify_demand)
            total_switchers = st.session_state.switcher_percentage * prev_df[prev_df["DEMAND"] == "HIGH"]["PEOPLE IN LINE"].sum()
            curr_df = day_df[day_df["TIME SLOT"] == ts].copy()
            curr_df["DEMAND"] = curr_df["WAIT TIME"].apply(classify_demand)
            num_low_demand = curr_df[curr_df["DEMAND"] == "LOW"].shape[0]
            switchers_per_low_attraction = total_switchers / num_low_demand if num_low_demand > 0 else 0
            if not row.empty:
                current_wait = row["WAIT TIME"].values[0]
                current_demand = classify_demand(current_wait)
                if current_demand == "LOW":
                    new_value = original_inside + switchers_per_low_attraction
                else:
                    new_value = original_inside
            else:
                new_value = None
            new_inside_list.append(new_value)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(time_labels_inside, original_inside_list, label="PEOPLE INSIDE ATTRACTION", marker="o")
    ax2.plot(time_labels_inside, new_inside_list, label="NEW PEOPLE INSIDE ATTRACTION", marker="o")
    ax2.set_xlabel("Time Slot")
    ax2.set_ylabel("People Count")
    ax2.set_title(f"People Inside {selected_attraction} Over the Day")
    ax2.legend()
    ax2.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

# -------------------------------
# PAGE: Predicting Park Attendance
# -------------------------------
elif page == "Predicting Park Attendance":
    st.markdown("<h1 style='text-align: center; color: #2C3E50;'>Predicting Park Attendance</h1>", unsafe_allow_html=True)

    # Attendance data from 2022
    file = "data_pp_merged_smaller.csv"
    df_file = pd.read_csv(file)
    df_2022 = df_file[df_file["WORK_DATE"] >= "2022-01-01"]
    df_2022["WORK_DATE"] = pd.to_datetime(df_2022["WORK_DATE"])
    average_attendance_2022 = df_2022["attendance"].mean()
    high_attendance = average_attendance_2022 + (average_attendance_2022 * 0.05)
    low_attendance = average_attendance_2022 - (average_attendance_2022 * 0.05)

    # Data for Predicted Attendance
    file_attendance_predicted = "predicted_future_attendance_august (1).csv"
    df_attendance_predicted = pd.read_csv(file_attendance_predicted)
    df_attendance_predicted["date"] = pd.to_datetime(df_attendance_predicted["date"])

    st.sidebar.header("Attendance Overview")
    st.write(f"### Average Attendance in 2022: {round(average_attendance_2022)}")
    default_date = datetime.date(2022, 8, 1)
    date = st.sidebar.date_input("Select a date for prediction:", value=default_date)
    date = pd.to_datetime(date)

    predicted_attendance = df_attendance_predicted[df_attendance_predicted["date"] == date]["ets_forecast"]
    if not predicted_attendance.empty:
        predicted_attendance = round(predicted_attendance.iloc[0])
    else:
        predicted_attendance = round(average_attendance_2022)
    st.markdown(f"### Predicted Attendance: {predicted_attendance}")

    if predicted_attendance <= low_attendance:
        attendance_level = "Low Attendance"
        color = "green"
    elif predicted_attendance >= high_attendance:
        attendance_level = "High Attendance"
        color = "red"
    else:
        attendance_level = "Average Attendance"
        color = "orange"

    prediction_start_date = "2022-08-01"
    df_2022["date"] = pd.to_datetime(df_2022["date"])
    df_attendance_predicted["date"] = pd.to_datetime(df_attendance_predicted["date"])
    history_start_date = pd.to_datetime(prediction_start_date) - pd.Timedelta(days=21)
    df_history = df_2022[(df_2022["date"] >= history_start_date) & (df_2022["date"] < prediction_start_date)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_history["date"], df_history["attendance"], color="blue", linewidth=2, label="Historical Attendance")
    ax.plot(df_attendance_predicted["date"], df_attendance_predicted["ets_forecast"], color="green", linewidth=2, label="Predicted Attendance")
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Attendance", fontsize=14)
    ax.set_title("Attendance Trends: Historical vs. Predicted", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    times_fine = np.linspace(9, 19, num=50)
    times_hours = np.arange(9, 20, 1)
    min_attendance = 1000
    skewness = 3
    log_rise = np.log(times_fine - 9 + 1)
    log_rise /= log_rise.max()
    skewed_distribution = skewnorm.pdf(times_fine, a=skewness, loc=10.5, scale=2)
    attendance_pattern = log_rise * skewed_distribution
    attendance_pattern = attendance_pattern / attendance_pattern.max()
    attendance_pattern = attendance_pattern * (predicted_attendance - min_attendance) + min_attendance
    attendance_hourly = np.interp(times_hours, times_fine, attendance_pattern)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times_fine, attendance_pattern, label=f"Estimated Attendance for {date.date()}", color='green', linewidth=2)
    ax.fill_between(times_fine, attendance_pattern, alpha=0.3, color='green')
    ax.set_xticks(range(9, 20))
    ax.set_xticklabels([f"{h}:00" for h in range(9, 20)])
    ax.set_xlabel("Time of Day", fontsize=14)
    ax.set_ylabel("Estimated Attendance", fontsize=14)
    ax.set_title(f"Estimated Park Attendance for {date.date()}", fontsize=16)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown(f"<h2 style='color:{color};'>{attendance_level}</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("Picture1.jpeg", width=300)
    with col2:
        if attendance_level == "Low Attendance":
            st.markdown("<h2 style='font-size: 36px;'>2 Rides in order to win a Fast Pass</h2>", unsafe_allow_html=True)
            number_rides = 2
        elif attendance_level == "High Attendance":
            st.markdown("<h2 style='font-size: 36px;'>4 Rides in order to win a Fast Pass</h2>", unsafe_allow_html=True)
            number_rides = 4
        else:
            st.markdown("<h2 style='font-size: 36px;'>3 Rides in order to win a Fast Pass</h2>", unsafe_allow_html=True)
            number_rides = 3

# -------------------------------
# PAGE: Predicting Waiting Times
# -------------------------------
elif page == "Home": 
    logo1_url = "https://upload.wikimedia.org/wikipedia/fr/thumb/8/86/Logo_CentraleSup%C3%A9lec.svg/1200px-Logo_CentraleSup%C3%A9lec.svg.png"
    logo2_url = "https://eleven-strategy.com/wp-content/uploads/2022/12/logo-eleven-vert.png"
    st.markdown("<h1 style='text-align: center;'>Prediction Dashboard of PortAventura Park</h1>", unsafe_allow_html=True)
    st.write("")
    st.header(":rocket: Get Started !")
    st.write("")
    st.write("Choose a page from the sidebar:")
    st.markdown("""
    - **Attendance Prediction**: Get information on predicted attendance for a specific day.
    - **Waiting Time Prediction**: Get information on predicted waiting times for each rides.
    - **Simulation**: Visualize impact of BIG Strategy.
    """)
    st.write("<br>", unsafe_allow_html=True)
    st.header(":telephone_receiver: Contact Us")
    st.markdown("""
    - **Léo BLANC**: leo.blanc@student-cs.fr
    - **Juliette LACROIX**: juliette.lacroix@student-cs.fr
    - **Hugo LE CORRE**: hugo.le-corre@student-cs.fr
    - **Philippe MIRANDA-JEAN**: philippe.mirandajean@student-cs.fr
    - **Théo ROSSI**: theo.rossi@student-cs.fr
    - **Elizaveta VASILEVA**: elizaveta.vasileva@tudent-cs.fr
    """)
    
    st.write("<br><br><br>", unsafe_allow_html=True)

    col1, col2 = st.columns([0.4, 0.4])
    with col1:
        # Center the first image using st.markdown with HTML and URL
        st.markdown(f"<div style='text-align: center;'><img src='{logo1_url}' width='200'></div>", unsafe_allow_html=True)

    with col2:
        # Center the second image using st.markdown with HTML and URL
        st.markdown(f"<div style='text-align: center;'><img src='{logo2_url}' width='200'></div>", unsafe_allow_html=True)

elif page == "Predicting Waiting Times":
    st.markdown("<h1 style='text-align: center; color: #2C3E50;'>Predicting Waiting Times</h1>", unsafe_allow_html=True)
    st.sidebar.header("Waiting Time Overview")
    
    # Load data for waiting times
    file_wait = "data_pp_merged_smaller.csv"
    df_file_wait = pd.read_csv(file_wait)
    df_2022_wait = df_file_wait[df_file_wait["WORK_DATE"] >= "2022-01-01"]
    df_2022_wait["WORK_DATE"] = pd.to_datetime(df_2022_wait["WORK_DATE"])
    wait_time_av_2022 = df_2022_wait["WAIT_TIME_MAX"].mean()
    st.write(f"### Average Waiting Time in 2022: **{round(wait_time_av_2022)}** minutes")
    
    file_future = "future_predictions_1.csv"
    df_waiting = pd.read_csv(file_future)
    df_waiting["date"] = pd.to_datetime(df_waiting["date"])
    df_waiting["hour"] = df_waiting["date"].dt.hour
    df_waiting["day"] = df_waiting["date"].dt.date
    
    default_date_wait = datetime.date(2022, 8, 2)
    selected_date_wait = st.sidebar.date_input("Select a date for prediction:", value=default_date_wait)
    selected_date_wait = pd.to_datetime(selected_date_wait)
    
    threshold = st.sidebar.slider("Set the threshold for the waiting time (minutes)", 
                                  min_value=0, max_value=120, value=30, step=5)
    
    list_rides = df_waiting["ENTITY_DESCRIPTION_SHORT"].unique().tolist()
    list_rides.sort()
    st.sidebar.title("Select a Ride to See Its Waiting Time")
    selected_ride = st.sidebar.selectbox("Select a Ride", list_rides)
    
    df_ride_selected = df_waiting[df_waiting["ENTITY_DESCRIPTION_SHORT"] == selected_ride]
    df_ride_selected_day = df_ride_selected[df_ride_selected["day"] == selected_date_wait.date()]
    
    if not df_ride_selected_day.empty:
        wait_time_av_day = df_ride_selected_day["Predicted_WAIT_TIME_MAX"].mean()
        st.write(f"### Average Waiting Time for {selected_ride} on {selected_date_wait.date().strftime('%B %d, %Y')}: {round(wait_time_av_day, 2)} minutes")
    else:
        st.write(f"### No data available for {selected_ride} on {selected_date_wait.date().strftime('%B %d, %Y')}.")
    
    if not df_ride_selected_day.empty:
        df_ride_selected_day['Hour'] = df_ride_selected_day['date'].dt.hour
        df_hourly_wait = df_ride_selected_day.groupby('Hour')['Predicted_WAIT_TIME_MAX'].mean().reset_index()
    
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_hourly_wait['Hour'], df_hourly_wait['Predicted_WAIT_TIME_MAX'], marker='o', linestyle='-', color='b')
        ax.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold ({threshold} min)")
        ax.set_title(f"Maximum Waiting Time for {selected_ride} on {selected_date_wait.date().strftime('%B %d, %Y')}", fontsize=14)
        ax.set_xlabel('Hour of the Day', fontsize=12)
        ax.set_ylabel('Max Wait Time (minutes)', fontsize=12)
        ax.grid(True)
        plt.xticks(df_hourly_wait['Hour'])
        plt.xticks(rotation=45)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write(f"### No data available for {selected_ride} on {selected_date_wait.date().strftime('%B %d, %Y')}.")
    
    rides_above_threshold = []
    rides_below_threshold = []
    
    for ride in list_rides:
        df_ride = df_waiting[df_waiting["ENTITY_DESCRIPTION_SHORT"] == ride]
        df_ride_day = df_ride[df_ride["day"] == selected_date_wait.date()]
        if not df_ride_day.empty:
            avg_wait_time = round(df_ride_day["Predicted_WAIT_TIME_MAX"].mean())
            hours_above = df_ride_day[df_ride_day["Predicted_WAIT_TIME_MAX"] > threshold]["date"].dt.hour.unique().tolist()
            hours_below = df_ride_day[df_ride_day["Predicted_WAIT_TIME_MAX"] <= threshold]["date"].dt.hour.unique().tolist()
            if hours_above:
                rides_above_threshold.append({
                    "Ride": ride,
                    "Avg Wait Time (min)": avg_wait_time,
                    "Hours Above Threshold": ", ".join(map(str, sorted(hours_above))),
                })
            if hours_below:
                rides_below_threshold.append({
                    "Ride": ride,
                    "Avg Wait Time (min)": avg_wait_time,
                    "Hours Below Threshold": ", ".join(map(str, sorted(hours_below))),
                })
    
    df_above_threshold = pd.DataFrame(rides_above_threshold).sort_values(by="Avg Wait Time (min)", ascending=False)
    df_below_threshold = pd.DataFrame(rides_below_threshold).sort_values(by="Avg Wait Time (min)", ascending=False)
    
    st.write("### Rides Exceeding the Threshold")
    st.dataframe(df_above_threshold)
    st.write("### Rides Below the Threshold")
    st.dataframe(df_below_threshold)
