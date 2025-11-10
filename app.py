import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Set the Streamlit page configuration
st.set_page_config(
    page_title="Simple Marks Predictor (Streamlit)",
    page_icon="🎓",
    layout="wide",
)

# --- 1. Load and Cache Data ---
# Use st.cache_data to load data only once
DATA_FILE = "student_scores.csv"

@st.cache_data
def load_data():
    """Loads the student scores data."""
    try:
        data = pd.read_csv(DATA_FILE)
        # Ensure 'Hours' and 'Scores' columns are correctly formatted
        data.dropna(inplace=True)
        # Verify the presence of the required columns
        if 'Hours' not in data.columns or 'Scores' not in data.columns:
            st.error("Error: The CSV must contain 'Hours' and 'Scores' columns.")
            st.stop()
        return data
    except FileNotFoundError:
        st.error(f"Error: '{DATA_FILE}' not found. Please ensure the file is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

data = load_data()

# --- 2. Train and Cache Model ---
# Use st.cache_resource to train the model only once
@st.cache_resource
def train_model(data):
    """Trains the Simple Linear Regression model."""
    # Using 'Hours' as feature (X) and 'Scores' as target (y)
    X = data['Hours'].values.reshape(-1, 1)
    y = data['Scores'].values
    
    # Split data (using random_state=0 for reproducibility as in the source file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate R-squared for display
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    
    return model, r2

model, r2_val = train_model(data)

# --- 3. Streamlit App Layout ---

st.title("🎓 Student Score Predictor (Simple Linear Model)")
st.markdown("Predict the student's score based on the **Study Hours** using a simple Linear Regression model trained on your uploaded data.")

st.markdown("---")

# --- 4. Sidebar for Input ---
st.sidebar.header("Input Parameters")
# Find min and max hours from the dataset for a relevant slider range
min_hours = float(data['Hours'].min())
max_hours = float(data['Hours'].max())

# Interactive slider for the user
hours_input = st.sidebar.slider(
    'Study Hours per Day/Week', 
    min_value=min_hours, 
    max_value=max_hours + 1.0, 
    value=5.0, 
    step=0.1
)

# --- 5. Prediction Logic ---
# Predict the score
prediction = model.predict(np.array([[hours_input]]))[0]

# Clip the score between 0 and 100 for realistic output
final_score = np.clip(prediction, 0, 100)
rounded_score = round(final_score, 2)

# --- 6. Results and Visuals ---

col1, col2 = st.columns(2)

with col1:
    st.subheader("Predicted Score")
    # Display the result in a large metric box
    st.metric(
        label=f"Score for {hours_input} Hours", 
        value=f"{rounded_score}%", 
        delta_color="off",
        help="The prediction is rounded to two decimal places and capped at 100%."
    )
    
    st.markdown(f"### Model Metrics")
    st.info(f"R² Score (Model Fit): **{r2_val:.4f}** (A score close to 1.0 indicates a very strong fit.)")
    st.write(f"Equation: Score = **{model.coef_[0]:.4f}** * Hours + **{model.intercept_:.4f}**")


with col2:
    st.subheader("Regression Visualization")

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 1. Plot the actual data points
    ax.scatter(data['Hours'], data['Scores'], color='purple', label='Actual Data')
    
    # 2. Plot the regression line
    # Use the min and max of the entire 'Hours' column for the plotting range
    X_plot_min = data['Hours'].min()
    X_plot_max = data['Hours'].max() + 1.0 # Extend slightly past max for better visual
    
    X_plot = np.linspace(X_plot_min, X_plot_max, 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    ax.plot(X_plot, y_plot, color='red', linestyle='--', label='Regression Line')
    
    # 3. Plot the specific prediction point
    ax.scatter(hours_input, final_score, color='gold', s=200, label='Your Prediction', edgecolors='black', zorder=5)

    ax.set_title('Study Hours vs. Student Scores')
    ax.set_xlabel('Hours Studied')
    ax.set_ylabel('Final Score (%)')
    ax.legend()
    ax.grid(True)
    
    # Display the plot in Streamlit
    st.pyplot(fig)


st.markdown("---")

st.header("Raw Data")
st.dataframe(data, use_container_width=True)
