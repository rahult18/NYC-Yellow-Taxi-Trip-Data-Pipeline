import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from plotly.subplots import make_subplots


color_palette = ['#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#59a14f', '#edc949', '#af7aa1', '#ff9da7', '#9c755f', '#bab0ab']

st.set_page_config(page_title="NYC Yellow Taxi Trip EDA Dashboard", page_icon="🚕", layout="wide")

# Database connection
engine = create_engine('postgresql://rahul:@localhost:5432/nyc')

# Fetch data from the database
@st.cache_data
def fetch_data():
    query = "SELECT * FROM cleaned_tripdata"
    return pd.read_sql(query, engine)

df = fetch_data()

# Load taxi zone lookup data
zone_lookup = pd.read_csv("/Users/rahul/PycharmProjects/nycTaxi/deployment/taxi_zone_lookup.csv")
zone_lookup['LocationID'] = zone_lookup['LocationID'].astype(int)
df['pulocationid'] = df['pulocationid'].astype(int)
df['dolocationid'] = df['dolocationid'].astype(int)
df = pd.merge(df, zone_lookup, left_on='pulocationid', right_on='LocationID', how='left')
df = pd.merge(df, zone_lookup, left_on='dolocationid', right_on='LocationID', how='left')
df.rename(columns={"Zone_x": "pickup_zone", "Zone_y": "dropoff_zone"}, inplace=True)

# Sidebar filters
st.sidebar.title("Filters")
year = st.sidebar.selectbox("Select Year", options=sorted(df['tpep_pickup_datetime'].dt.year.unique()))
month = st.sidebar.selectbox("Select Month", options=sorted(df['tpep_pickup_datetime'].dt.month.unique()))

# Filter data based on user selections
filtered_df = df[(df['tpep_pickup_datetime'].dt.year == year) & (df['tpep_pickup_datetime'].dt.month == month)]

# Main content
st.title("NYC Yellow Taxi Trip EDA Dashboard")
st.write(f"Showing data for {year}-{month:02d}")

# Metrics
total_trips = len(filtered_df)
total_passengers = filtered_df['passenger_count'].sum()
total_revenue = filtered_df['total_amount'].sum()
avg_trip_duration = filtered_df['trip_duration'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trips", f"{total_trips:,}")
col2.metric("Total Passengers", f"{total_passengers:,}")
col3.metric("Total Revenue", f"${total_revenue:,.2f}")
col4.metric("Avg. Trip Duration", f"{avg_trip_duration:.2f} min")

# Top pickup and dropoff locations
st.subheader("Top Pickup and Dropoff Locations")
col1, col2 = st.columns(2)

with col1:
    top_pickups = filtered_df['pickup_zone'].value_counts().head(10)
    fig_pickups = px.bar(top_pickups, x=top_pickups.index, y=top_pickups.values,
                         title="Top 10 Pickup Locations",
                         labels={'x': 'Location', 'y': 'Number of Pickups'},
                         color=top_pickups.index,
                         color_discrete_sequence=color_palette)
    fig_pickups.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_pickups, use_container_width=True)

with col2:
    top_dropoffs = filtered_df['dropoff_zone'].value_counts().head(10)
    fig_dropoffs = px.bar(top_dropoffs, x=top_dropoffs.index, y=top_dropoffs.values,
                          title="Top 10 Dropoff Locations",
                          labels={'x': 'Location', 'y': 'Number of Dropoffs'},
                          color=top_dropoffs.index,
                          color_discrete_sequence=color_palette)
    fig_dropoffs.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_dropoffs, use_container_width=True)

# Average fare by time of day
st.subheader("Average Fare by Time of Day")

def categorize_time(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

filtered_df['time_of_day'] = filtered_df['tpep_pickup_datetime'].dt.hour.apply(categorize_time)

avg_fare_by_time = filtered_df.groupby('time_of_day')['fare_amount'].mean().reindex(['Morning', 'Afternoon', 'Evening', 'Night'])

fig_fare = px.bar(avg_fare_by_time, x=avg_fare_by_time.index, y=avg_fare_by_time.values,
                   labels={'x': 'Time of Day', 'y': 'Average Fare ($)'},
                   title="Average Fare by Time of Day",
                   color=avg_fare_by_time.index,
                   color_discrete_sequence=color_palette)
st.plotly_chart(fig_fare, use_container_width=True)

# Payment type distribution
st.subheader("Payment Type Distribution")
payment_type_map = {
    1: "Credit card",
    2: "Cash",
    3: "No charge",
    4: "Dispute",
    5: "Unknown",
    6: "Voided trip"
}

filtered_df['payment_type'] = pd.to_numeric(filtered_df['payment_type'], errors='coerce').fillna(0).astype(int)
filtered_df['payment_type_name'] = filtered_df['payment_type'].map(payment_type_map).fillna("Other")

payment_type_counts = filtered_df['payment_type_name'].value_counts().reset_index()
payment_type_counts.columns = ['Payment Type', 'Count']

fig_payment = px.pie(
    payment_type_counts,
    values='Count',
    names='Payment Type',
    hole=0.3,
    color_discrete_sequence=color_palette
)

fig_payment.update_traces(textposition='inside', textinfo='percent+label')
fig_payment.update_layout(
    legend_title_text='Payment Type',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_payment, use_container_width=True)

# Trip duration distribution
st.subheader("Trip Duration Distribution")

Q1 = filtered_df['trip_duration'].quantile(0.25)
Q3 = filtered_df['trip_duration'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_no_outliers = filtered_df[(filtered_df['trip_duration'] >= lower_bound) &
                             (filtered_df['trip_duration'] <= upper_bound)]

fig_duration = px.histogram(df_no_outliers, x='trip_duration', nbins=50,
                            labels={'trip_duration': 'Trip Duration (minutes)'},
                            color_discrete_sequence=color_palette)

fig_duration.update_layout(
    xaxis_title="Trip Duration (minutes)",
    yaxis_title="Count",
    showlegend=False
)

mean_duration = df_no_outliers['trip_duration'].mean()
fig_duration.add_vline(x=mean_duration, line_dash="dash", line_color="red",
                       annotation_text=f"Mean: {mean_duration:.2f} min",
                       annotation_position="top right")

st.plotly_chart(fig_duration, use_container_width=True)

st.write(f"**Trip Duration Statistics:**")
col1, col2, col3 = st.columns(3)
col1.metric("Mean", f"{mean_duration:.2f} min")
col2.metric("Median", f"{df_no_outliers['trip_duration'].median():.2f} min")
col3.metric("Standard Deviation", f"{df_no_outliers['trip_duration'].std():.2f} min")

# Hourly trip count
st.subheader("Hourly Trip Count")
hourly_trips = filtered_df.groupby(filtered_df['tpep_pickup_datetime'].dt.hour).size()
fig_hourly = px.bar(hourly_trips, x=hourly_trips.index, y=hourly_trips.values,
                    labels={'x': 'Hour of Day', 'y': 'Number of Trips'},
                    color=hourly_trips.index,
                    color_discrete_sequence=color_palette)
fig_hourly.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
st.plotly_chart(fig_hourly, use_container_width=True)

# Improved Tip Visualization
st.subheader("Tip Analysis by Day of Week")

filtered_df['day_of_week'] = filtered_df['tpep_pickup_datetime'].dt.dayofweek
filtered_df['day_name'] = filtered_df['tpep_pickup_datetime'].dt.day_name()

tip_data = filtered_df.groupby('day_of_week').agg({
    'tip_amount': 'mean',
    'total_amount': 'mean'
}).reset_index()

tip_data['tip_percentage'] = (tip_data['tip_amount'] / tip_data['total_amount']) * 100
tip_data['day_name'] = pd.Categorical(tip_data['day_of_week'].map({0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}), categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
tip_data = tip_data.sort_values('day_name')

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Bar(x=tip_data['day_name'], y=tip_data['tip_amount'], name="Avg Tip Amount", marker_color=color_palette[0]),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=tip_data['day_name'], y=tip_data['tip_percentage'], name="Tip Percentage", marker_color=color_palette[1], mode='lines+markers'),
    secondary_y=True,
)

fig.update_layout(
    title_text="Average Tip Amount and Percentage by Day of Week",
    xaxis_title="Day of Week",
    barmode='group',
    legend=dict(y=1.1, x=0.5, xanchor='center', orientation='h')
)

fig.update_yaxes(title_text="Average Tip Amount ($)", secondary_y=False)
fig.update_yaxes(title_text="Tip Percentage (%)", secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

st.write("**Tip Statistics:**")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Highest Avg Tip", f"${tip_data['tip_amount'].max():.2f}", tip_data.loc[tip_data['tip_amount'].idxmax(), 'day_name'])
col2.metric("Lowest Avg Tip", f"${tip_data['tip_amount'].min():.2f}", tip_data.loc[tip_data['tip_amount'].idxmin(), 'day_name'])
col3.metric("Highest Tip %", f"{tip_data['tip_percentage'].max():.2f}%", tip_data.loc[tip_data['tip_percentage'].idxmax(), 'day_name'])
col4.metric("Lowest Tip %", f"{tip_data['tip_percentage'].min():.2f}%", tip_data.loc[tip_data['tip_percentage'].idxmin(), 'day_name'])

overall_avg_tip = filtered_df['tip_amount'].mean()
overall_tip_percentage = (filtered_df['tip_amount'].sum() / filtered_df['total_amount'].sum()) * 100

st.write("**Overall Tipping Behavior:**")
col1, col2 = st.columns(2)
col1.metric("Overall Average Tip", f"${overall_avg_tip:.2f}")
col2.metric("Overall Tip Percentage", f"{overall_tip_percentage:.2f}%")

# Average Speed by Hour
st.subheader("Average Speed by Hour")
filtered_df['speed'] = filtered_df['trip_distance'] / (filtered_df['trip_duration'] / 60)  # mph
avg_speed_by_hour = filtered_df.groupby(filtered_df['tpep_pickup_datetime'].dt.hour)['speed'].mean()
fig_speed = px.line(avg_speed_by_hour, x=avg_speed_by_hour.index, y=avg_speed_by_hour.values,
                    labels={'x': 'Hour of Day', 'y': 'Average Speed (mph)'},
                    title="Average Speed by Hour of Day",
                    color_discrete_sequence=color_palette)
fig_speed.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
st.plotly_chart(fig_speed, use_container_width=True)

# Fare Breakdown
st.subheader("Fare Breakdown")
fare_components = ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'congestion_surcharge', 'airport_fee']
fare_breakdown = filtered_df[fare_components].mean()
fig_fare_breakdown = px.pie(values=fare_breakdown.values, names=fare_breakdown.index, color_discrete_sequence=color_palette)
st.plotly_chart(fig_fare_breakdown, use_container_width=True)

# Top 10 Routes
st.subheader("Top 10 Routes")
filtered_df['route'] = filtered_df['pickup_zone'] + ' to ' + filtered_df['dropoff_zone']
top_routes = filtered_df['route'].value_counts().head(10)
fig_routes = px.bar(top_routes, x=top_routes.index, y=top_routes.values,
                    labels={'x': 'Route', 'y': 'Number of Trips'},
                    color=top_routes.index,
                    color_discrete_sequence=color_palette)
fig_routes.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_routes, use_container_width=True)