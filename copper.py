import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import streamlit as st
import re
import pickle

st.set_page_config(layout="wide")

st.write("""
<div style='text-align:center'>
    <h1 style='color:#009999;'>Industrial_Copper_Modeling </h1>
</div>
""", unsafe_allow_html=True)

def about_the_developer():
    st.header("About the Developer")
    st.subheader("Sethumadhavan V")
    st.markdown("""
    📧 Email: sethumadhavanvelu2002@example.com  
    📞 Phone: 9159299878  
    [LinkedIn](https://www.linkedin.com/in/sethumadhavan-v-b84890257/)  
    [GitHub](https://github.com/SETHU0010/Industrial_Copper_Modeling.git)
    """, unsafe_allow_html=True)

def skills_take_away():
    st.header("Skills Take Away From This Project")
    st.caption("Python Scripting")
    st.caption("Data Preprocessing")
    st.caption("EDA")
    st.caption("Streamlit")
        
def objective():
    st.header("Objective")
    st.write("Address skewness and noisy data in copper industry sales and pricing. Predict selling prices and classify leads (status: WON or LOST)")

def features():
    st.header("Features")
    st.write("Unique transaction IDs, dates, quantities, customer details, countries, statuses, item types, applications, thickness, width, material and product references, delivery dates, and selling prices.")

def workflow():
    st.header("Workflow")
    st.write("1. Data Understanding")
    st.write("2. Data Preprocessing ")
    st.write("3. Exploratory Data Analysis (EDA)")
    st.write("4. Feature Engineering")
    st.write("5. Model Building and Evaluation")
    st.write("6. Streamlit GUI Development")
    st.write("7. Deployment and Model Persistence")

def prerequisites():
    st.header("Prerequisites")
    st.write("Before using the application, ensure you have the following prerequisites set up:")
    st.write("1. Python programming")
    st.write("2. Data preprocessing and EDA techniques")
    st.write("3. Machine learning basics")
        
def required_python_libraries():
    st.header("Required Python Libraries")
    st.write("The following Python libraries are required for the project:")
    libraries = ["pandas", "Numpy", "Matplotlib","Seaborn","Scikit-learn","Pickle", "streamlit"]
    st.write(libraries)

def Approach():
    st.header("Approach")
    st.write("1. Identify variable types and distributions")
    st.write("2. Handle missing values and outliers")
    st.write("3. Treat skewness and encode categorical variables")
    st.write("4. Visualize outliers and skewness")
    st.write("5. Create and drop features as needed")
    st.write("6. Train and evaluate regression and classification models")
    st.write("7. Develop a Streamlit app for interactive predictions")


def Learning_Outcomes():
    st.header("Learning Outcomes")
    st.write("1. Proficiency in Python and data analysis libraries")
    st.write("2. Data preprocessing and EDA skills")
    st.write("3. Machine learning modeling and evaluation")
    st.write("4. Streamlit app development")
    st.write("5. Understanding manufacturing domain challenges")

def main():
    # Main layout
    col1, col2 = st.columns(2)

    with col1:
        st.header("Navigation")
        options = ["About the Developer", "Skills take away From This Project", "Objective", "Features", 
                   "Workflow", "Prerequisites", "Required Python Libraries", "Approach", "Learning Outcomes"]
        choice = st.radio("Go to", options)

    with col2:
        if choice == "About the Developer":
            about_the_developer()
        elif choice == "Skills take away From This Project":
            skills_take_away()
        elif choice == "Objective":
            objective()
        elif choice == "Features":
            features()
        elif choice == "Workflow":
            workflow()
        elif choice == "Prerequisites":
            prerequisites()
        elif choice == "Required Python Libraries":
            required_python_libraries()
        elif choice == "Approach":
            Approach()
        elif choice == "Learning Outcomes":
            Learning_Outcomes()
if __name__ == "__main__":
    main()

# Options for dropdown menus
status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
product_options = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def validate_inputs(inputs):
    pattern = "^(?:\d+|\d*\.\d+)$"
    for i in inputs:
        if not re.match(pattern, i):
            return False, i
    return True, None

def prepare_input(inputs, ohe_transformer, scaler):
    ohe_features = ohe_transformer.transform(inputs[:, [7]]).toarray()
    inputs = np.concatenate((inputs[:, [0, 1, 2, 3, 4, 5, 6]], ohe_features), axis=1)
    inputs_scaled = scaler.transform(inputs)
    return inputs_scaled
# Add tabs for prediction
tab1, tab2 = st.tabs(["🔮 Predict Selling Price", "🔍 Predict Lead Status"])

# Options for dropdown menus
status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
product_options = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def validate_inputs(inputs):
    pattern = "^(?:\d+|\d*\.\d+)$"
    for i in inputs:
        if not re.match(pattern, i):
            return False, i
    return True, None

def prepare_input(inputs, ohe_transformer, scaler):
    ohe_features = ohe_transformer.transform(inputs[:, [7]]).toarray()
    inputs = np.concatenate((inputs[:, [0, 1, 2, 3, 4, 5, 6]], ohe_features), axis=1)
    inputs_scaled = scaler.transform(inputs)
    return inputs_scaled

# Tab 1: Predict Selling Price
with tab1:
    st.header("🔮 Predict Selling Price")
    with st.form("price_form"):
        col1, col2 = st.columns(2)

        with col1:
            status = st.selectbox("Status", status_options, key=1)
            item_type = st.selectbox("Item Type", item_type_options, key=2)
            country = st.selectbox("Country", sorted(country_options), key=3)
            application = st.selectbox("Application", sorted(application_options), key=4)
            product_ref = st.selectbox("Product Reference", product_options, key=5)
        
        with col2:
            st.write("**Note:** Min & Max values are for reference; you can enter any value within the range.")
            quantity_tons = st.text_input("Quantity Tons (Min: 611728, Max: 1722207579)")
            thickness = st.text_input("Thickness (Min: 0.18, Max: 400)")
            width = st.text_input("Width (Min: 1, Max: 2990)")
            customer = st.text_input("Customer ID (Min: 12458, Max: 30408185)")
            submit_button = st.form_submit_button(label="Predict Selling Price")
        
        if submit_button:
            inputs_valid, invalid_input = validate_inputs([quantity_tons, thickness, width, customer])
            if not inputs_valid:
                if len(invalid_input) == 0:
                    st.write("Please enter a valid number. No spaces allowed.")
                else:
                    st.write(f"You have entered an invalid value: {invalid_input}")
            else:
                model = load_model("model.pkl")
                scaler = load_model("scaler.pkl")
                ohe_transformer = load_model("t.pkl")
                be_transformer = load_model("s.pkl")
                
                new_sample = np.array([[np.log(float(quantity_tons)), application, np.log(float(thickness)), float(width), country, float(customer), int(product_ref), item_type, status]])
                new_sample_prepared = prepare_input(new_sample, ohe_transformer, scaler)
                
                predicted_price = model.predict(new_sample_prepared)[0]
                st.write(f'## :green[Predicted Selling Price:] ${predicted_price:.2f}')

# Tab 2: Predict Status
with tab2:
    st.header("🔍 Predict Lead Status")
    with st.form("status_form"):
        col1, col2 = st.columns(2)

        with col1:
            cquantity_tons = st.text_input("Quantity Tons (Min: 611728, Max: 1722207579)")
            cthickness = st.text_input("Thickness (Min: 0.18, Max: 400)")
            cwidth = st.text_input("Width (Min: 1, Max: 2990)")
            ccustomer = st.text_input("Customer ID (Min: 12458, Max: 30408185)")
            cselling = st.text_input("Selling Price (Min: 1, Max: 100001015)")
        
        with col2:
            citem_type = st.selectbox("Item Type", item_type_options, key=21)
            ccountry = st.selectbox("Country", sorted(country_options), key=31)
            capplication = st.selectbox("Application", sorted(application_options), key=41)
            cproduct_ref = st.selectbox("Product Reference", product_options, key=51)
            csubmit_button = st.form_submit_button(label="Predict Lead Status")
        
        if csubmit_button:
            inputs_valid, invalid_input = validate_inputs([cquantity_tons, cthickness, cwidth, ccustomer, cselling])
            if not inputs_valid:
                if len(invalid_input) == 0:
                    st.write("Please enter a valid number. No spaces allowed.")
                else:
                    st.write(f"You have entered an invalid value: {invalid_input}")
            else:
                model = load_model("cmodel.pkl")
                scaler = load_model("cscaler.pkl")
                ohe_transformer = load_model("ct.pkl")
                
                new_sample = np.array([[int(ccustomer), ccountry, capplication, float(cwidth), int(cproduct_ref), np.log(float(cselling)), np.log(float(cquantity_tons)), np.log(float(cthickness)), citem_type]])
                new_sample_prepared = prepare_input(new_sample, ohe_transformer, scaler)
                
                predicted_status = model.predict(new_sample_prepared)
                if predicted_status == 1:
                    st.write('## :green[The Status is Won]')
                else:
                    st.write('## :red[The Status is Lost]')
