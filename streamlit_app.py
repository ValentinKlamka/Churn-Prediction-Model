import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

def load_pickles(model_pickle_path, label_encoder_pickle_path):
    """
    Loading the saved pickles with model and label encoder
    """
    model_pickle_opener = open(model_pickle_path, "rb")
    model = pickle.load(model_pickle_opener)

    label_encoder_pickle_opener = open(label_encoder_pickle_path, "rb")
    label_encoder_dict = pickle.load(label_encoder_pickle_opener)

    return model, label_encoder_dict

def pre_process_data(df, label_encoder_dict):
    """
    Apply pre-processing steps from before to new data
    """
    for col in df.columns:
        if col in list(label_encoder_dict.keys()):
            # accessing the column's label encoder via
            # column name as key
            column_le = label_encoder_dict[col]
            # applying fitted label encoder to the data
            df.loc[:, col] = column_le.transform(df.loc[:, col])
        else:
            continue
    return df

def make_predictions(processed_df, model):
    prediction = model.predict(processed_df)
    return prediction

def generate_predictions(test_df):
    model_pickle_path = "churn_prediction_model.pkl"
    label_encoder_pickle_path = "churn_prediction_label_encoders.pkl"

    model, label_encoder_dict = load_pickles(model_pickle_path,
                                             label_encoder_pickle_path)

    processed_df = pre_process_data(test_df, label_encoder_dict)
    prediction = make_predictions(processed_df, model)

    return prediction


if __name__ == "__main__":
    st.title("Telcos Vorhersage zur Abwanderung")
    st.subheader("Geben Sie die Daten ihres Kunden ein, dessen Verhalten Sie vorhersagen wollen:")

    # making customer data inputs
    gender = st.selectbox("Welches Geschlecht hat der Kunde:",
                          ['Weiblich', "Männlich"])
    if gender == "Weiblich":
        gender = "Female"
    else:
        gender = "Male"
    senior_citizen_input = st.selectbox("Ist der Kunde in Rente?:",
                                        ["Ja", "Nein"])
    if senior_citizen_input == "Ja":
        senior_citizen = 1
    else:
        senior_citizen = 0

    partner = st.selectbox("Hat der Kunde einen Lebensgefährten?:",
                          ["Ja", "Nein"])
    if partner == "Ja":
        partner = "Yes"
    else:
        partner = "No"
    dependents = st.selectbox("Hat der Kunde Angehörige?:",
                             ["Ja", "Nein"])
    if dependents == "Ja":
        dependents = "Yes"
    else:
        dependents = "No"
    tenure = st.slider("Wie viele Monate ist der Kunde im Unternehmen?:",
                       min_value=0, max_value=72, value=42)
    phone_service = st.selectbox("Hat der Kunde einen Telefonservice?:",
                                 ["Ja", "Nein"])
    if phone_service == "Ja":
        phone_service = "Yes"
    else:
        phone_service = "No"  
    multiple_lines = st.selectbox("Hat der Kunde mehrere Leitungen?:",
                                 ["Ja", "Nein", "Kein Telefonservice"])
    if multiple_lines == "Ja":
        multiple_lines = "Yes"
    elif multiple_lines == "Nein":
        multiple_lines = "No"
    else:
        multiple_lines = "No phone service"
    internet_service = st.selectbox("Hat der Kunde Internetdienst?:",
                                 ["Nein", "DSL", "Fiber Optic"])
    if internet_service == "DSL":
        internet_service = "DSL"
    elif internet_service == "Fiber Optic":
        internet_service = "Fiber optic"
    else:
        internet_service = "No"
    online_security = st.selectbox("Hat der Kunde Online Security?:",
                                 ["Ja", "Nein", "Kein Internet Service"])
    if online_security == "Ja":
        online_security = "Yes"
    elif online_security == "Nein":
        online_security = "No"
    else:
        online_security = "No internet service"

    online_backup = st.selectbox("Hat der Kunde Online Backup?:",
                                 ["Ja", "Nein", "Kein Internet Service"])
    if online_backup == "Ja":
        online_backup = "Yes"
    elif online_backup == "Nein":
        online_backup = "No"
    else:   
        online_backup = "No internet service"
    device_protection = st.selectbox("Hat der Kunde Geräteschutz?:",
                                    ["Ja", "Nein", "Kein Internet Service"])
    if device_protection == "Ja":
        device_protection = "Yes"
    elif device_protection == "Nein":
        device_protection = "No"
    else:
        device_protection = "No internet service"

    tech_support = st.selectbox("Hat der Kunde Technischen Support?:",
                                ["Ja", "Nein", "Kein Internet Service"])
    if tech_support == "Ja":
        tech_support = "Yes"
    elif tech_support == "Nein":
        tech_support = "No"
    else:
        tech_support = "No internet service"
    streaming_tv = st.selectbox("Hat der Kunde Streaming TV?:",
                                 ["Ja", "Nein", "Kein Internet Service"])
    if streaming_tv == "Ja":
        streaming_tv = "Yes"
    elif streaming_tv == "Nein":
        streaming_tv = "No"
    else:
        streaming_tv = "No internet service"
    streaming_movies = st.selectbox("Hat der Kunde Streaming Movies?:",
                                     ["Ja", "Nein", "Kein Internet Service"])
    if streaming_movies == "Ja":
        streaming_movies = "Yes"
    elif streaming_movies == "Nein":
        streaming_movies = "No"
    else:
        streaming_movies = "No internet service"
    contract = st.selectbox("Welches Vertragsmodell hat der Kunde?:",
                            ["Monatlich", "Zwei Jahre", "Ein Jahr"])
    if contract == "Monatlich":
        contract = "Month-to-month"
    elif contract == "Zwei Jahre":
        contract = "Two year"
    else:
        contract = "One year"
    paperless_billing = st.selectbox("Hat der Kunde eine Papierlose Rechnung?:",
                                        ["Ja", "Nein"])
    if  paperless_billing == "Ja":
        paperless_billing = "Yes"
    else:
        paperless_billing = "No"
    payment_method = st.selectbox("Wie bezahlt der Kunde?:",
                                ["Electronic Check", "Mailed Check", "Bank Transfer (Automatic)",
                                "Credit Card (Automatic)"])
    if payment_method == "Electronic Check":
        payment_method = "Electronic check"
    elif payment_method == "Mailed Check":
        payment_method = "Mailed check"
    elif payment_method == "Bank Transfer (Automatic)":
        payment_method = "Bank transfer (automatic)"
    else:
        payment_method = "Credit card (automatic)"
    monthly_charges = st.slider("Wie hoch sind die monatlichen Kosten?:",
                                min_value=0.0, max_value=118.0, value=42.0)
    total_charges = st.slider("Wie hoch sind die Gesamtkosten?:",
                            min_value=0.0, max_value=8660.0, value=42.0)

        # creating a dictionary of the inputs


    input_dict = {"gender": gender,
                  "SeniorCitizen": senior_citizen,
                  "Partner": partner,
                  "Dependents": dependents,
                  "tenure": tenure,
                  "PhoneService": phone_service,
                  "MultipleLines": multiple_lines,
                  "InternetService": internet_service,
                  "OnlineSecurity": online_security,
                  "OnlineBackup": online_backup,
                  "DeviceProtection": device_protection,
                  "TechSupport": tech_support,
                  "StreamingTV": streaming_tv,
                  "StreamingMovies": streaming_movies,
                  "Contract": contract,
                  "PaperlessBilling": paperless_billing,
                  "PaymentMethod": payment_method,
                  "MonthlyCharges": monthly_charges,
                  "TotalCharges": total_charges}

    input_data = pd.DataFrame([input_dict])

    if st.button("Vorhersagen!"):
        pred = generate_predictions(input_data)
        if bool(pred):
            st.error("Kunde wird abwandern!")
        else:
            st.success("Kunde wird wahrscheinlich bleiben!")