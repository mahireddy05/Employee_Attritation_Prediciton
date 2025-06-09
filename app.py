import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, roc_curve, auc,
                             precision_recall_curve)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Configuration
st.set_page_config(page_title="Employee Attrition Analytics", layout="wide")


# Caching
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df


@st.cache_data
def preprocess_data(df):
    df = df.drop(['EmployeeCount', 'StandardHours', 'Over18'], axis=1)
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].astype('category')
    return df


# Sidebar Configuration
with st.sidebar:
    st.header("Data Configuration")
    uploaded_file = st.file_uploader("Upload HR Dataset (CSV)", type="csv")

    if uploaded_file:
        df = load_data(uploaded_file)
        df = preprocess_data(df)

        st.header("Model Settings")
        models = st.multiselect(
            "Select Models",
            ['Logistic Regression', 'SVM', 'Random Forest', 'KNN', 'Naive Bayes',
             'XGBoost', 'LightGBM', 'Gradient Boosting', 'CatBoost', 'AdaBoost',
             'Neural Network'],
            default=['XGBoost', 'Random Forest', 'LightGBM']
        )

        test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2)
        random_state = st.number_input("Random State", 42)

        st.header("Advanced Features")
        feature_importance = st.checkbox("Show SHAP Feature Importance", True)
        hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning")

        if hyperparameter_tuning:
            n_estimators = st.slider("Number of Estimators", 50, 500, 100)
            learning_rate = st.slider("Learning Rate", 0.001, 1.0, 0.1)

# Main Application
if uploaded_file:
    st.title("Employee Attrition Predictive Analytics Dashboard")

    # Data Overview Section
    st.header("Data Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df.head(), height=250)
    with col2:
        st.write("Dataset Shape:", df.shape)
        st.write("Attrition Distribution:", df['Attrition'].value_counts().to_dict())

    # Enhanced EDA Visualizations
    st.header("Exploratory Data Analysis")
    with st.expander("Advanced Data Visualizations", expanded=True):
        # Correlation Heatmap
        st.subheader("Feature Correlation Analysis")
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)

        st.markdown("""
        **Insight:** Identify strong correlations between features and attrition.
        - Red: Positive correlation, Blue: Negative correlation
        - Focus on features with absolute correlation > 0.3
        """)

        # Refine the department-wise attrition analysis for better accuracy and visualization
        st.subheader("Department-wise Attrition Analysis")
        department_attrition = df.groupby(['Department', 'Attrition']).size().unstack(fill_value=0)
        department_attrition_percentage = department_attrition.div(department_attrition.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(10, 6))
        department_attrition_percentage.plot(kind='bar', stacked=True, ax=ax, color=['skyblue', 'salmon'])
        ax.set_title("Attrition Distribution Across Departments")
        ax.set_ylabel("Percentage")
        ax.set_xlabel("Department")
        ax.legend(title="Attrition", labels=['No', 'Yes'])
        st.pyplot(fig)

        st.markdown("""
        **Insight:** Compare attrition rates across departments.
        - Departments with >15% attrition need immediate attention
        - Helps prioritize intervention strategies
        """)

    # Data Preprocessing
    selected_features = ['Age', 'MonthlyIncome', 'JobSatisfaction',
                         'WorkLifeBalance', 'YearsAtCompany', 'OverTime',
                         'Department', 'JobRole', 'MaritalStatus']
    X = df[selected_features]
    y = df['Attrition'].cat.codes

    categorical_features = X.select_dtypes(include=['category']).columns
    numerical_features = X.select_dtypes(include=['number']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Model Configuration
    models_dict = {
        'Logistic Regression': LogisticRegression(),
        'SVM': SVC(probability=True),
        'Random Forest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'CatBoost': CatBoostClassifier(silent=True),
        'AdaBoost': AdaBoostClassifier(),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
    }

    # Model Training
    st.header("Model Training & Evaluation")
    results = []

    for model_name in models:
        with st.spinner(f"Training {model_name}..."):
            # Handle hyperparameters
            if hyperparameter_tuning:
                if model_name in ['XGBoost', 'LightGBM', 'CatBoost']:
                    models_dict[model_name].set_params(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate
                    )
                elif model_name == 'Neural Network':
                    models_dict[model_name].set_params(
                        hidden_layer_sizes=(n_estimators, n_estimators // 2)
                    )

            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', models_dict[model_name])
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            results.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc,
                'Pipeline': pipeline,
                'CM': cm,
                'fpr': fpr,
                'tpr': tpr,
                'y_proba': y_proba
            })

    # Display Results
    display_df = pd.DataFrame([{
        'Model': r['Model'],
        'Accuracy': r['Accuracy'],
        'Precision': r['Precision'],
        'Recall': r['Recall'],
        'F1 Score': r['F1 Score'],
        'ROC AUC': r['ROC AUC']
    } for r in results]).set_index('Model')

    pipelines = {r['Model']: r['Pipeline'] for r in results}

    st.dataframe(display_df.style.format("{:.2%}").highlight_max(axis=0),
                 use_container_width=True)

    # Model Diagnostics
    st.header("Model Performance Analysis")
    col1, col2 = st.columns(2)

    with col1:
        selected_model = st.selectbox("Select Model for Details", display_df.index)
        model_idx = display_df.index.get_loc(selected_model)

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(results[model_idx]['CM'], annot=True, fmt='d', ax=ax,
                    cmap='Blues', cbar=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

    with col2:
        st.subheader("ROC Curve")
        fpr = results[model_idx]['fpr']
        tpr = results[model_idx]['tpr']
        roc_auc = results[model_idx]['ROC AUC']

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                 mode='lines',
                                 name=f'{selected_model} (AUC = {roc_auc:.2f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                 mode='lines',
                                 line=dict(dash='dash'),
                                 name='Random'))
        fig.update_layout(title='Receiver Operating Characteristic',
                          xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate')
        st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    if feature_importance:
        st.header("Feature Importance Analysis")
        selected_model = st.selectbox("Select Model for Interpretation", display_df.index)
        pipeline = pipelines[selected_model]

        try:
            explainer = shap.Explainer(pipeline.named_steps['classifier'])
            X_processed = preprocessor.transform(X)
            shap_values = explainer.shap_values(X_processed)

            st.subheader("SHAP Summary Plot")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_processed,
                              feature_names=preprocessor.get_feature_names_out())
            st.pyplot(fig)

            st.markdown("""
            **Interpretation Guide:**
            - Features pushing prediction to the right increase attrition risk
            - Color shows feature value (red=high, blue=low)
            - Overlapping points indicate cluster patterns
            """)
        except Exception as e:
            st.warning(f"Feature importance not available for {selected_model}: {str(e)}")

    # Prediction Interface
    st.header("🎯 Employee Attrition Prediction")
    st.markdown("""
Use this tool to predict the attrition risk of an employee based on key features.
- **Select a model** from the dropdown below.
- **Input employee details** to get a prediction.
""")

    selected_model = st.selectbox("🔍 Select Prediction Model", display_df.index)
    pipeline = pipelines[selected_model]

    with st.form("prediction_form"):
        st.markdown("### 📝 Employee Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 18, 65, 30, help="Enter the employee's age.")
            monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000, help="Enter the employee's monthly income.")
            department = st.selectbox("Department", df['Department'].cat.categories, help="Select the employee's department.")
        with col2:
            job_role = st.selectbox("Job Role", df['JobRole'].cat.categories, help="Select the employee's job role.")
            marital_status = st.selectbox("Marital Status", df['MaritalStatus'].cat.categories, help="Select the employee's marital status.")
            overtime = st.selectbox("Overtime", ['Yes', 'No'], help="Does the employee work overtime?")
        with col3:
            job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3, help="Rate the employee's job satisfaction.")
            work_life_balance = st.slider("Work-Life Balance (1-4)", 1, 4, 3, help="Rate the employee's work-life balance.")
            years_at_company = st.number_input("Years at Company", 0, 40, 3, help="Enter the number of years the employee has been at the company.")

        if st.form_submit_button("🚀 Predict Attrition Risk"):
            input_data = pd.DataFrame([
                [
                    age, monthly_income, job_satisfaction, work_life_balance,
                    years_at_company, overtime, department, job_role, marital_status
                ]
            ], columns=selected_features)

            try:
                prediction = pipeline.predict(input_data)[0]
                probability = pipeline.predict_proba(input_data)[0][1]

                st.markdown(f"### 🎯 Prediction: {'**High Risk**' if prediction == 1 else '**Low Risk**'}")
                st.metric("Attrition Probability", f"{probability:.1%}")

                # Enhanced visualization of prediction probability
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["Attrition", "No Attrition"], [probability, 1 - probability], color=['#FF6F61', '#6EC4E8'])
                ax.set_title("Attrition Probability Distribution", fontsize=14)
                ax.set_ylabel("Probability", fontsize=12)
                ax.set_xticks([0, 1], labels=["Attrition", "No Attrition"], fontsize=12)
                st.pyplot(fig)

                if probability > 0.7:
                    st.error("⚠️ **High risk employee** - Recommend retention actions.")
                elif probability > 0.4:
                    st.warning("⚠️ **Moderate risk** - Monitor closely.")
                else:
                    st.success("✅ **Low risk** - No immediate action needed.")

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

    # Model Download
    st.header("Model Management")
    best_model = display_df['F1 Score'].idxmax()
    buffer = BytesIO()
    joblib.dump(pipelines[best_model], buffer)
    buffer.seek(0)

    st.download_button(
        label=f"Download Best Model ({best_model})",
        data=buffer,
        file_name=f"best_model_{best_model}.pkl",
        mime="application/octet-stream"
    )

else:
    st.title("Employee Attrition Prediction Dashboard")
    st.markdown("""
    ### Welcome to HR Analytics Dashboard
    1. Upload your employee dataset (CSV format)
    2. Configure models in the left sidebar
    3. Explore data insights
    4. Train and evaluate models
    5. Predict attrition risk for individual employees
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/3771/3771401.png", width=200)