
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from app import preprocess_data, predict_new_data, generate_recommendations

# Load models and other necessary files
@st.cache_resource
def load_artifacts():
    try:
        log_model = joblib.load("logistic_model.pkl")
        lin_model = joblib.load("linear_model.pkl")
        scaler = joblib.load("scaler.pkl")
        kmeans = joblib.load("kmeans_model.pkl")
        cluster_scaler = joblib.load("cluster_scaler.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        return log_model, lin_model, scaler, kmeans, cluster_scaler, feature_columns
    except FileNotFoundError:
        st.error("Model files not found. Please run 'app.py' to train and save models first.")
        return None, None, None, None, None, None

log_model, lin_model, scaler, kmeans, cluster_scaler, feature_columns = load_artifacts()

if log_model is not None:
    st.title("AI Learning Analytics - Student Performance Prediction")
    st.write("Enter student data to predict performance and get recommendations.")

    # CSV Upload Section
    st.markdown("---")
    st.header("Bulk Prediction Upload")
    st.write("Upload a CSV file containing student data to generate predictions for multiple students at once.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df_upload.head())
            
            if st.button("Generate Predictions for Uploaded File"):
                with st.spinner("Processing..."):
                    # We need to preprocess the uploaded data
                    # The app.py functions expect a certain format.
                    # Let's assume the CSV has the same columns as the form inputs (or similar to raw data)
                    
                    # We'll use predict_new_data which calls preprocess_data
                    # preprocess_data handles "ParentEduc", "WklyStudyHours" cleaning etc.
                    # It also calculates AverageScore if Math/Reading/Writing are present.
                    
                    results_upload_df = predict_new_data(df_upload, log_model, lin_model, scaler, kmeans, cluster_scaler)
                    results_upload_df = generate_recommendations(results_upload_df)
                    
                    st.success("Batch Analysis Complete!")
                    
                    # Display results
                    st.subheader("Results")
                    
                    # Filter columns to show relevant info + predictions
                    # We want to keep input data + predictions + recommendations and Learner Category
                    
                    # Identify columns to drop if they exist - only drop Cluster ID
                    cols_to_drop = ["Cluster"]
                    results_display = results_upload_df.drop(columns=[c for c in cols_to_drop if c in results_upload_df.columns], errors='ignore')

                    st.dataframe(results_display)
                    
                    # Download button
                    csv = results_display.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name='student_predictions.csv',
                        mime='text/csv',
                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")

    st.markdown("---")
    st.header("Individual Prediction")

    # Input Form
    with st.form("student_data_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["female", "male"])
            ethnic_group = st.selectbox("Ethnic Group", ["group A", "group B", "group C", "group D", "group E"])
            parent_educ = st.selectbox("Parent's Education", ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
            lunch_type = st.selectbox("Lunch Type", ["standard", "free/reduced"])
            test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])

        with col2:
            parent_marital_status = st.selectbox("Parent Marital Status", ["married", "single", "widowed", "divorced"])
            practice_sport = st.selectbox("Practice Sport", ["regularly", "sometimes", "never"])
            is_first_child = st.selectbox("Is First Child", ["yes", "no"])
            nr_siblings = st.number_input("Number of Siblings", min_value=0, max_value=10, value=1)
            transport_means = st.selectbox("Transport Means", ["school_bus", "private"])
            wkly_study_hours = st.selectbox("Weekly Study Hours", ["< 5", "5 - 10", "> 10"])

        st.subheader("Current Scores (For Analysis)")
        col3, col4, col5 = st.columns(3)
        with col3:
            math_score = st.number_input("Math Score", min_value=0, max_value=100, value=70)
        with col4:
            reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=70)
        with col5:
            writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=70)

        submitted = st.form_submit_button("Predict & Recommend")

    if submitted:
        # Create input DataFrame
        input_data = {
            "Gender": [gender],
            "EthnicGroup": [ethnic_group],
            "ParentEduc": [parent_educ],
            "LunchType": [lunch_type],
            "TestPrep": [test_prep],
            "ParentMaritalStatus": [parent_marital_status],
            "PracticeSport": [practice_sport],
            "IsFirstChild": [is_first_child],
            "NrSiblings": [nr_siblings],
            "TransportMeans": [transport_means],
            "WklyStudyHours": [wkly_study_hours],
            "MathScore": [math_score],
            "ReadingScore": [reading_score],
            "WritingScore": [writing_score]
        }
        
        df_new = pd.DataFrame(input_data)

        # Handle potential formatting differences for WklyStudyHours matches app.py logic
        # app.py expects: "<5", "5-10", ">10"
        # Streamlit inputs: "< 5", "5 - 10", "> 10"
        # The preprocess_data function strips spaces, so "< 5" becomes "<5", "5 - 10" becomes "5-10"
        # So we can pass it as is, provided preprocess_data handles it correctly.
        
        try:
            # Predict
            # Using the imported predict_new_data function might be tricky if it relies on global scope or specific file paths
            # But the logic is self-contained in the function except for feature_columns loading
            # We already loaded the models, so we can pass them.
            
            # Note: app.py's predict_new_data loads feature_columns inside. 
            # We can update app.py to accept feature_columns as argument OR just trust it works if pkl is in same dir.
            # However, looking at app.py, predict_new_data calls preprocess_data
            
            # Make sure we don't have circular dependencies or issues calling app.py functions
            
            # Let's call the function from app.py
            # BUT: app.py loads feature_columns.pkl using joblib.load("feature_columns.pkl")
            # We need to make sure we are in the right directory.
            
            results_df = predict_new_data(df_new, log_model, lin_model, scaler, kmeans, cluster_scaler)
            results_df = generate_recommendations(results_df)
            
            st.success("Analysis Complete!")
            
            st.subheader("Predictions")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pass/Fail Prediction", "Pass" if results_df["Predicted_PassFail"][0] == 1 else "Fail")
            with col2:
                st.metric("Predicted Average Score", f"{results_df['Predicted_AverageScore'][0]:.2f}")
            with col3:
                st.metric("Learner Category", results_df["Learner Category"][0])
            
            st.subheader("Recommendation")
            st.info(results_df["Recommendation"][0])

            # Visualization Section
            st.markdown("---")
            st.subheader("Visualizations")
            
            try:
                # Load and preprocess original data for context
                from app import load_data, preprocess_data
                raw_df = load_data("./Data/raw/Student_Performance.csv")
                processed_df = preprocess_data(raw_df)
                
                # 1. Score Comparison
                st.write("### Score Comparison")
                avg_score_dataset = processed_df["AverageScore"].mean()
                student_score = results_df['Predicted_AverageScore'][0]
                
                fig1, ax1 = plt.subplots(figsize=(8, 4))
                sns.barplot(x=["Dataset Average", "Student Predicted"], y=[avg_score_dataset, student_score], ax=ax1, palette=["grey", "blue"])
                ax1.set_ylabel("Average Score")
                ax1.set_ylim(0, 100)
                ax1.bar_label(ax1.containers[0], fmt='%.1f')
                st.pyplot(fig1)

                # 2. Cluster Visualization
                st.write("### Learner Category Cluster Visualization")
                
                # We need to assign clusters to the dataset to color it
                # Using the logic from app.py/train_kmeans roughly
                cluster_features_df = processed_df[["AverageScore", "WklyStudyHours"]]
                cluster_scaled_df = cluster_scaler.transform(cluster_features_df)
                processed_df["Cluster"] = kmeans.predict(cluster_scaled_df)
                
                # Map clusters to labels (Logic from app.py)
                cluster_centers = kmeans.cluster_centers_[:, 0]
                sorted_indices = np.argsort(cluster_centers)
                cluster_labels_map = {
                    sorted_indices[0]: "At Risk",
                    sorted_indices[1]: "Average",
                    sorted_indices[2]: "High Performer"
                }
                processed_df["Learner Category"] = processed_df["Cluster"].map(cluster_labels_map)

                # Prepare student data point
                student_wkly_study_hours = results_df["WklyStudyHours"][0] # This is numeric 3, 7, 12 from preprocess
                # Note: results_df is already processed by predict_new_data calling preprocess_data internally?
                # Let's check app.py: predict_new_data calls preprocess_data(df_new). 
                # preprocess_data maps strings to 3, 7, 12. 
                # So results_df["WklyStudyHours"] is numeric.

                feature_columns = joblib.load("feature_columns.pkl") # Reloading to be safe or use loaded one? 
                # We loaded feature_columns in load_artifacts return.
                
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                
                # Plot background dataset
                sns.scatterplot(
                    data=processed_df, 
                    x="WklyStudyHours", 
                    y="AverageScore", 
                    hue="Learner Category", 
                    hue_order=["At Risk", "Average", "High Performer"],
                    alpha=0.6,
                    palette="viridis",
                    ax=ax2
                )
                
                # Plot student point
                ax2.scatter(
                    x=student_wkly_study_hours, 
                    y=student_score, 
                    color='red', 
                    s=200, 
                    edgecolors='black', 
                    marker='X', 
                    label='You',
                    zorder=10
                )
                
                ax2.set_title("Study Hours vs. Average Score")
                ax2.legend()
                st.pyplot(fig2)

            except Exception as e:
                st.error(f"Could not load visualizations: {e}")


        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.warning("Please ensure model files (*.pkl) are present in the directory.")
