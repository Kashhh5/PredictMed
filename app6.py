import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

def load_image(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load models
diabetes_model = pickle.load(open(r'diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(r'heart_disease_model.sav', 'rb'))

bg_image = load_image(r'doc.webp')

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'PREDICT MED',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Help Section', 'Analysis'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'info-circle', 'bar-chart'],
        default_index=0
    )

# CSS for background image and tooltip
st.markdown(f"""
    <style>
    body {{
        background-image: url(data:image/png;base64,{bg_image});
        background-size: cover;
        background-position: no-repeat;
        color: white;
        opacity: 0.9;
    }}
    .tooltip {{
        position: relative;
        display: inline-block;
        cursor: pointer;
    }}
    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 5px;
        padding: 5px 0;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }}
    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    </style>
    """, unsafe_allow_html=True)

# Function to save input data to CSV
def save_input_data(data, filename="user_inputs.csv"):
    file_exists = os.path.isfile(filename)
    df = pd.DataFrame([data])
    if file_exists:
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='w', header=True, index=False)

# Diabetes Prediction Page
# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    # Gender Selection
    gender = st.radio("Select Gender *", ['Female', 'Male'], help="Choose the gender of the person.")

    # Form columns for inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Username = st.text_input("Enter your username:")

        if gender == 'Female':
            Pregnancies = st.text_input('Number of Pregnancies *', help="Number of times pregnant; relates to hormonal changes.")
        else:
            Pregnancies = 0
            st.text("Pregnancies: Not Applicable for Males")
             
    with col2:
        Glucose = st.text_input('Glucose Level *', help="Plasma glucose level (mg/dL). Higher levels suggest diabetes.")
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value *', help="Blood pressure in arteries (mm Hg).")
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value *', help="Thickness of triceps skin fold (mm).")
    
    with col2:
        Insulin = st.text_input('Insulin Level *', help="Insulin level in blood (µU/mL).")
    
    with col3:
        BMI = st.text_input('BMI value *', help="Body Mass Index (kg/m²).")
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value *', help="Genetic predisposition score.")
    
    with col2:
        Age = st.text_input('Age of the Person *', help="Age factor for diabetes risk.")

    diab_diagnosis = ''

    # Prediction button
    if st.button('Diabetes Test Result'):
        try:
            user_input = [
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age),
            ]
            diab_prediction = diabetes_model.predict([user_input])
            st.session_state['diab_diagnosis'] = 'Positive' if diab_prediction[0] == 1 else 'Negative'
        except ValueError:
            st.session_state['diab_diagnosis'] = "Please ensure all inputs are filled and numeric."
        st.success(st.session_state['diab_diagnosis'])
    st.success(diab_diagnosis)

     
    if st.button('Save Before CSV'):
        if Username:
            data = {
                "Username": Username,
                "Pregnancies": Pregnancies,
                "Glucose": Glucose,
                "BloodPressure": BloodPressure,
                "SkinThickness": SkinThickness,
                "Insulin": Insulin,
                "BMI": BMI,
                "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
                "Age": Age,
                "Diabetes Diagnosis": st.session_state.get('diab_diagnosis', '')  # Add diagnosis to the data
            }
            df = pd.DataFrame([data])
            file_name = f"{Username}_diabetes_before.csv"
            df.to_csv(file_name, index=False)
            st.success(f"Results saved as {file_name}")

            df_before = pd.DataFrame([data])
            
            # Generate CSV for download
            csv_before = df_before.to_csv(index=False)
            st.download_button(
                label="Download Before CSV",
                data=csv_before,
                file_name=f"{Username}_diabetes_before.csv",
                mime='text/csv'
            )
        else:
            st.error("Please enter a username to save results.")

    if st.button('Save After CSV'):
        if Username:
            data = {
                "Username": Username,
                "Pregnancies": Pregnancies,
                "Glucose": Glucose,
                "BloodPressure": BloodPressure,
                "SkinThickness": SkinThickness,
                "Insulin": Insulin,
                "BMI": BMI,
                "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
                "Age": Age,
                "Diabetes Diagnosis": st.session_state.get('diab_diagnosis', '')  # Add diagnosis to the data
            }
            df = pd.DataFrame([data])
            file_name = f"{Username}_diabetes_after.csv"
            df.to_csv(file_name, index=False)
            st.success(f"Results saved as {file_name}")

            df_after = pd.DataFrame([data])
            # Generate CSV for download
            csv_after = df_after.to_csv(index=False)
            st.download_button(
                label="Download After CSV",
                data=csv_after,
                file_name=f"{Username}_diabetes_after.csv",
                mime='text/csv'
            )
        else:
            st.write("Please enter a username to save results.")


elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    # Input fields for heart disease prediction
    col1, col2, col3 = st.columns(3)
    with col1:
        Username = st.text_input("Enter your username:")
        age = st.text_input('Age *', help="Age of the person.")
    with col2:
        sex = st.text_input('Sex *', help="Gender. 0-Female, 1-Male.")
    with col3:
        cp = st.text_input('Chest Pain types *', help="Types include angina, non-anginal pain, asymptomatic.")
    with col1:
        trestbps = st.text_input('Resting Blood Pressure *', help="")
    with col2:
        chol = st.text_input('Serum Cholesterol in mg/dl *')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl *')
    with col1:
        restecg = st.text_input('Resting ECG results *')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved *')
    with col3:
        exang = st.text_input('Exercise Induced Angina *')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise *')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment *')
    with col3:
        ca = st.text_input('Major vessels colored by fluoroscopy *')
    with col1:
        thal = st.text_input('Thalassemia *')

    # Button for Heart Disease Test Result
    # Initialize session state for heart_diagnosis
    heart_diagnosis =''
    # Prediction Button
    if st.button('Heart Disease Test Result'):
        try:
            user_input = [float(x) for x in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
            heart_prediction = heart_disease_model.predict([user_input])
            st.session_state['heart_diagnosis'] = 'Positive' if heart_prediction[0] == 1 else 'Negative'
        except ValueError:
            st.session_state['heart_diagnosis'] = "Please ensure all inputs are filled and numeric."
        st.success(st.session_state['heart_diagnosis'])

    # Save CSV Button
    if st.button('Save Heart Before CSV'):
        if Username:
            data = {
                "Username": Username,
                "Age": age,
                "Sex": sex,
                "Chest Pain Type": cp,
                "Resting BP": trestbps,
                "Cholesterol": chol,
                "Fasting Blood Sugar": fbs,
                "Resting ECG": restecg,
                "Max Heart Rate": thalach,
                "Exercise Induced Angina": exang,
                "Oldpeak": oldpeak,
                "Slope": slope,
                "CA": ca,
                "Thalassemia": thal,
                "Heart Diagnosis": st.session_state['heart_diagnosis']  # Use session state to retain the result
            }

            df = pd.DataFrame([data])
            file_name = f"{Username}_heart_before.csv"
            df.to_csv(file_name, index=False)
            st.success(f"Results saved as {file_name}")

            df_heart_before = pd.DataFrame([data])
        
        # Generate CSV for download
            csv_heart_before = df_heart_before .to_csv(index=False)
            st.download_button(
                label="Download Heart Before CSV",
                data=csv_heart_before,
                file_name=f"{Username}_heart_before.csv",
                mime='text/csv'
            )
        else:
            st.error("Please enter a username to save results.")
    

    if st.button('Save Heart After CSV'):
        if Username:
            # Prepare the data dictionary
            data = {
                "Username": Username,
                "Age": age,
                "Sex": sex,
                "Chest Pain Type": cp,
                "Resting BP": trestbps,
                "Cholesterol": chol,
                "Fasting Blood Sugar": fbs,
                "Resting ECG": restecg,
                "Max Heart Rate": thalach,
                "Exercise Induced Angina": exang,
                "Oldpeak": oldpeak,
                "Slope": slope,
                "CA": ca,
                "Thalassemia": thal,
                "Heart Diagnosis": st.session_state['heart_diagnosis'] 
            }
            df = pd.DataFrame([data])
            file_name = f"{Username}_heart_after.csv"
            df.to_csv(file_name, index=False)
            st.success(f"Results saved as {file_name}")

            df_heart_after = pd.DataFrame([data])
        
        # Generate CSV for download
            csv_heart_after = df_heart_after.to_csv(index=False)
            st.download_button(
                label="Download After CSV",
                data=csv_heart_after,
                file_name=f"{Username}_heart_after.csv",
                mime='text/csv'
            )
        else:
            st.error("Please enter a username to save results.")
        
if selected == 'Help Section':
    st.title("Help Section")
    st.write("""
    Welcome to the Health Assistant! Here you can predict the likelihood of various health conditions based on input data. 

    ### Sections:
    In this section, you can find details about each input field required for the disease prediction tests.

    **Diabetes Prediction:**

    - **Pregnancies**: Refers to the number of times a patient has been pregnant. Pregnancy can cause hormonal changes that may affect glucose regulation, potentially contributing to diabetes, particularly gestational diabetes (which occurs during pregnancy).

    - **Glucose Level**: Plasma glucose concentration (measured in mg/dL). Elevated glucose levels indicate hyperglycemia, a defining characteristic of diabetes.
        - Normal (Fasting): 70-99 mg/dL
        - Prediabetes (Fasting): 100-125 mg/dL
        - Diabetes (Fasting): 126 mg/dL or higher

    - **Blood Pressure**: The pressure in the arteries when the heart is at rest between beats (measured in mm Hg). High blood pressure often coexists with diabetes and is a significant risk factor for cardiovascular complications.
        - Normal: 60-80 mm Hg
        - Elevated: 80-89 mm Hg
        - Hypertension Stage 1: 90-99 mm Hg
        - Hypertension Stage 2: 100 mm Hg or higher

    - **Skin Thickness**: Measurement of the thickness of the triceps skin fold (in mm), used to estimate body fat. Higher skin thickness may suggest insulin resistance.

    - **Insulin Level**: The amount of insulin in the blood (measured in µU/mL) two hours after ingesting glucose.
        - Normal (Fasting): 2-25 µU/mL
        - Higher levels may indicate insulin resistance.

    - **BMI**: Body fat measure based on height and weight (kg/m²). Higher BMI correlates with obesity, a risk factor for Type 2 diabetes.
        - Underweight: <18.5
        - Normal: 18.5-24.9
        - Overweight: 25-29.9
        - Obese: 30 or higher

    - **Diabetes Pedigree Function**: Estimates diabetes likelihood based on family history.

    - **Age**: Risk factor for diabetes, especially significant after age 45.

    **Heart Disease Prediction:**

    - **Age**: Patient's age in years; older age increases risk.

    - **Sex**: Gender of the patient.
        - 1: Male
        - 0: Female
        - Risk is generally higher in men but increases for women post-menopause.

    - **Chest Pain Type**: Indicates chest pain type, an important heart disease marker.
        - 0: Typical angina
        - 1: Atypical angina
        - 2: Non-anginal pain
        - 3: Asymptomatic

    - **Resting Blood Pressure**: Patient's blood pressure upon hospital admission.
        - Normal: 120/80 mm Hg
        - Hypertension Stage 1: 130-139/80-89 mm Hg
        - Hypertension Stage 2: ≥140/90 mm Hg

    - **Serum Cholesterol**: Total cholesterol in blood.
        - Normal: <200 mg/dL
        - Borderline high: 200-239 mg/dL
        - High: ≥240 mg/dL

    - **Fasting Blood Sugar**: Whether fasting blood sugar >120 mg/dL.
        - 1: High
        - 0: Normal

    - **Resting ECG**: Results of resting electrocardiogram.
        - 0: Normal
        - 1: Abnormalities
        - 2: Probable left ventricular hypertrophy

    - **Max Heart Rate**: Maximum heart rate achieved during exercise.

    - **Exercise Angina**: Angina presence during exercise.
        - 1: Yes
        - 0: No

    - **ST Depression**: ST segment depression during exercise, indicating myocardial ischemia.

    - **Slope**: ST segment slope at peak exercise.
        - 0: Upsloping
        - 1: Flat
        - 2: Downsloping

    - **Major Vessels**: Number of major vessels (0-3) affected by fluoroscopy.

    - **Thalassemia**: Blood disorder.
        - 0: Normal
        - 1: Fixed defect
        - 2: Reversible defect

  """)
    
if selected == "Analysis":
    st.title("Patient Report Analysis")

    # Buttons for each condition
    st.subheader("Select Analysis Type")
    analysis_type = st.radio("Choose the Prediction Type for Analysis", 
                              options=["Diabetes Prediction", "Heart Disease Prediction"], 
                              help="Select the health condition for which you want to analyze Before and After reports.")

    # Separate file uploaders for "Before" and "After"
    st.subheader(f"Upload Reports for {analysis_type}")
    before_file = st.file_uploader(f"Upload 'Before' Report for {analysis_type} (CSV)", type="csv", key="before")
    after_file = st.file_uploader(f"Upload 'After' Report for {analysis_type} (CSV)", type="csv", key="after")

    if before_file and after_file:
        # Load CSV files
        before_data = pd.read_csv(before_file)
        after_data = pd.read_csv(after_file)

        # Display the uploaded data
        st.write(f"**Before Report - {analysis_type}**")
        st.dataframe(before_data.head())
        st.write(f"**After Report - {analysis_type}**")
        st.dataframe(after_data.head())

        # Check for common columns
        common_columns = list(set(before_data.columns).intersection(set(after_data.columns)))

        if common_columns:
            # Dropdown to select a column for comparison
            selected_column = st.selectbox(
                f"Select a column to analyze for {analysis_type}",
                common_columns,
                help="Choose a column common to both files for analysis."
            )

            # Dropdown to select chart type
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Bar Chart", "Line Chart", "KDE Plot", "Boxplot", "Scatter Plot"],
                help="Choose the type of chart for visualization."
            )

            # Separate Visualizations for "Before" and "After"
            st.subheader("Before Data Visualization")
            if chart_type == "Bar Chart":
                bar_data = pd.DataFrame({
                    "Index": range(len(before_data[selected_column])),
                    "Value": before_data[selected_column]
                })

                plt.figure(figsize=(10, 6))
                sns.barplot(data=bar_data, x="Index", y="Value", color="blue")
                plt.title(f"Bar Chart (Before): {selected_column}")
                plt.xlabel("Index")
                plt.ylabel(selected_column)
                st.pyplot(plt)

            elif chart_type == "Line Chart":
                plt.figure(figsize=(10, 6))
                plt.plot(before_data[selected_column], label="Before", marker='o', color='blue')
                plt.title(f"Line Chart (Before): {selected_column}")
                plt.xlabel("Index")
                plt.ylabel(selected_column)
                plt.legend()
                st.pyplot(plt)

            elif chart_type == "KDE Plot":
                plt.figure(figsize=(10, 6))
                sns.kdeplot(before_data[selected_column], label="Before", color='blue', shade=True)
                plt.title(f"KDE Plot (Before): {selected_column}")
                plt.xlabel(selected_column)
                plt.ylabel("Density")
                plt.legend()
                st.pyplot(plt)

            elif chart_type == "Boxplot":
                plt.figure(figsize=(8, 5))
                sns.boxplot(x=["Before"] * len(before_data[selected_column]), y=before_data[selected_column], palette="Set2")
                plt.title(f"Boxplot (Before): {selected_column}")
                plt.xlabel("Type")
                plt.ylabel(selected_column)
                st.pyplot(plt)

            elif chart_type == "Scatter Plot":
                plt.figure(figsize=(10, 6))
                plt.scatter(range(len(before_data[selected_column])), before_data[selected_column], label="Before", color='blue')
                plt.title(f"Scatter Plot (Before): {selected_column}")
                plt.xlabel("Index")
                plt.ylabel(selected_column)
                plt.legend()
                st.pyplot(plt)

            st.subheader("After Data Visualization")
            if chart_type == "Bar Chart":
                bar_data = pd.DataFrame({
                    "Index": range(len(after_data[selected_column])),
                    "Value": after_data[selected_column]
                })

                plt.figure(figsize=(10, 6))
                sns.barplot(data=bar_data, x="Index", y="Value", color="red")
                plt.title(f"Bar Chart (After): {selected_column}")
                plt.xlabel("Index")
                plt.ylabel(selected_column)
                st.pyplot(plt)

            elif chart_type == "Line Chart":
                plt.figure(figsize=(10, 6))
                plt.plot(after_data[selected_column], label="After", marker='o', color='red')
                plt.title(f"Line Chart (After): {selected_column}")
                plt.xlabel("Index")
                plt.ylabel(selected_column)
                plt.legend()
                st.pyplot(plt)

            elif chart_type == "KDE Plot":
                plt.figure(figsize=(10, 6))
                sns.kdeplot(after_data[selected_column], label="After", color='red', shade=True)
                plt.title(f"KDE Plot (After): {selected_column}")
                plt.xlabel(selected_column)
                plt.ylabel("Density")
                plt.legend()
                st.pyplot(plt)

            elif chart_type == "Boxplot":
                plt.figure(figsize=(8, 5))
                sns.boxplot(x=["After"] * len(after_data[selected_column]), y=after_data[selected_column], palette="Set3")
                plt.title(f"Boxplot (After): {selected_column}")
                plt.xlabel("Type")
                plt.ylabel(selected_column)
                st.pyplot(plt)

            elif chart_type == "Scatter Plot":
                plt.figure(figsize=(10, 6))
                plt.scatter(range(len(after_data[selected_column])), after_data[selected_column], label="After", color='red')
                plt.title(f"Scatter Plot (After): {selected_column}")
                plt.xlabel("Index")
                plt.ylabel(selected_column)
                plt.legend()
                st.pyplot(plt)

        else:
            st.warning("No common columns found between the 'Before' and 'After' reports.")
    else:
        st.info("Please upload both 'Before' and 'After' reports to proceed.")
