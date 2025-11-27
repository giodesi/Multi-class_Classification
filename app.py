"""
Multi-class Classification Web App
A generalized Streamlit application for multi-class classification on any CSV dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Multi-class Classification App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f7f9fc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("<div class='main-header'><h1>🎯 Multi-class Classification Web App</h1></div>", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'preprocessed_data' not in st.session_state:
    st.session_state['preprocessed_data'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None

# Sidebar for configuration
with st.sidebar:
    st.header("📁 Data Upload")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your dataset in CSV format"
    )
    
    if uploaded_file is not None:
        # Load data
        st.session_state['data'] = pd.read_csv(uploaded_file)
        st.success(f"✅ Data loaded successfully! Shape: {st.session_state['data'].shape}")
        
        # Target column selection
        st.header("🎯 Target Selection")
        columns = st.session_state['data'].columns.tolist()
        target_column = st.selectbox(
            "Select target column",
            columns,
            help="Choose the column you want to predict"
        )
        
        # Feature selection
        st.header("📊 Feature Selection")
        feature_columns = st.multiselect(
            "Select feature columns",
            [col for col in columns if col != target_column],
            default=[col for col in columns if col != target_column],
            help="Choose the features to use for training"
        )
        
        # Preprocessing options
        st.header("⚙️ Preprocessing Options")
        scale_features = st.checkbox("Standardize numerical features", value=True)
        encode_categorical = st.checkbox("One-hot encode categorical features", value=True)
        
        # Model configuration
        st.header("🤖 Model Configuration")
        
        # Classification algorithm selection (only Logistic Regression)
        classifier_type = "Logistic Regression"
        st.info("📌 Using Logistic Regression classifier")
        
        # Multi-class strategy
        multiclass_strategy = st.selectbox(
            "Select multi-class strategy",
            ["One-vs-Rest (OvR)", "One-vs-One (OvO)", "Auto"],
            help="Choose the strategy for handling multiple classes"
        )
        
        # Test size
        test_size = st.slider(
            "Test set size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Proportion of data to use for testing"
        )
        
        # Random state
        random_state = st.number_input(
            "Random state",
            value=42,
            help="Set random state for reproducibility"
        )

# Main content area
if st.session_state['data'] is not None:
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Data Overview", "🔍 EDA", "⚙️ Preprocessing", "🎯 Training", "📈 Results", "🔮 Predict New Data"])
    
    with tab1:
        st.header("Data Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", st.session_state['data'].shape[0])
        with col2:
            st.metric("Total Columns", st.session_state['data'].shape[1])
        with col3:
            st.metric("Missing Values", st.session_state['data'].isnull().sum().sum())
        
        # Display first few rows
        st.subheader("First 10 Rows")
        st.dataframe(st.session_state['data'].head(10))
        
        # Data types
        st.subheader("Data Types")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Column Information:**")
            info_df = pd.DataFrame({
                'Column': st.session_state['data'].columns,
                'Type': st.session_state['data'].dtypes.astype(str),  # Convert dtype to string
                'Non-Null Count': st.session_state['data'].count(),
                'Null Count': st.session_state['data'].isnull().sum()
            })
            st.dataframe(info_df)
        
        with col2:
            st.write("**Basic Statistics:**")
            st.dataframe(st.session_state['data'].describe())
    
    with tab2:
        st.header("Exploratory Data Analysis")
        
        if 'target_column' in locals():
            # Target distribution
            st.subheader("Target Variable Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                target_counts = st.session_state['data'][target_column].value_counts()
                ax.bar(target_counts.index.astype(str), target_counts.values, color='#667eea')
                ax.set_xlabel(target_column)
                ax.set_ylabel("Count")
                ax.set_title("Target Distribution")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                target_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90)
                ax.set_title("Target Distribution (Percentage)")
                ax.set_ylabel("")
                st.pyplot(fig)
            
            # Feature distributions
            st.subheader("Feature Distributions")
            
            if 'feature_columns' in locals() and len(feature_columns) > 0:
                # Numerical features
                numerical_features = st.session_state['data'][feature_columns].select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numerical_features) > 0:
                    st.write("**Numerical Features:**")
                    
                    # Create subplots for numerical features
                    n_cols = min(3, len(numerical_features))
                    n_rows = (len(numerical_features) + n_cols - 1) // n_cols
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
                    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
                    
                    for i, col in enumerate(numerical_features[:9]):  # Limit to 9 features for display
                        axes[i].hist(st.session_state['data'][col].dropna(), bins=30, color='#764ba2', alpha=0.7)
                        axes[i].set_title(f"Distribution of {col}")
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel("Frequency")
                    
                    # Hide unused subplots
                    for i in range(len(numerical_features), len(axes)):
                        axes[i].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Categorical features
                categorical_features = st.session_state['data'][feature_columns].select_dtypes(include=['object']).columns.tolist()
                
                if len(categorical_features) > 0:
                    st.write("**Categorical Features:**")
                    
                    for col in categorical_features[:5]:  # Limit to 5 features for display
                        st.write(f"*{col}*")
                        value_counts = st.session_state['data'][col].value_counts()
                        st.bar_chart(value_counts)
    
    with tab3:
        st.header("Data Preprocessing")
        
        if 'target_column' in locals() and 'feature_columns' in locals():
            
            if st.button("🔄 Apply Preprocessing"):
                with st.spinner("Preprocessing data..."):
                    
                    # Create a copy of the data
                    processed_data = st.session_state['data'][feature_columns + [target_column]].copy()
                    
                    # Handle missing values
                    st.write("**Handling Missing Values:**")
                    missing_counts = processed_data.isnull().sum()
                    
                    # Store training data statistics for later use in predictions
                    training_medians = {}
                    training_modes = {}
                    
                    if missing_counts.sum() > 0:
                        # Fill numerical columns with median
                        numerical_cols = processed_data.select_dtypes(include=[np.number]).columns
                        for col in numerical_cols:
                            if col != target_column:
                                median_val = processed_data[col].median()
                                training_medians[col] = median_val
                                if processed_data[col].isnull().any():
                                    processed_data[col].fillna(median_val, inplace=True)
                        
                        # Fill categorical columns with mode
                        categorical_cols = processed_data.select_dtypes(include=['object']).columns
                        for col in categorical_cols:
                            if col != target_column:
                                mode_val = processed_data[col].mode()[0] if len(processed_data[col].mode()) > 0 else 'unknown'
                                training_modes[col] = mode_val
                                if processed_data[col].isnull().any():
                                    processed_data[col].fillna(mode_val, inplace=True)
                        
                        st.success("✅ Missing values handled")
                    else:
                        # Still store statistics even if no missing values
                        numerical_cols = processed_data.select_dtypes(include=[np.number]).columns
                        for col in numerical_cols:
                            if col != target_column:
                                training_medians[col] = processed_data[col].median()
                        
                        categorical_cols = processed_data.select_dtypes(include=['object']).columns
                        for col in categorical_cols:
                            if col != target_column and len(processed_data[col].mode()) > 0:
                                training_modes[col] = processed_data[col].mode()[0]
                        
                        st.info("No missing values found")
                    
                    # Store training statistics in session state
                    st.session_state['training_medians'] = training_medians
                    st.session_state['training_modes'] = training_modes
                    
                    # Separate features and target
                    X = processed_data.drop(columns=[target_column])
                    y = processed_data[target_column]
                    
                    # Encode target variable if it's categorical
                    if y.dtype == 'object':
                        label_encoder = LabelEncoder()
                        y = label_encoder.fit_transform(y)
                        st.session_state['label_encoder'] = label_encoder
                        st.success(f"✅ Target variable encoded: {len(np.unique(y))} classes")
                    
                    # Identify numerical and categorical columns
                    numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
                    
                    # Standardize numerical features
                    if scale_features and len(numerical_columns) > 0:
                        st.write("**Standardizing Numerical Features:**")
                        scaler = StandardScaler()
                        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
                        st.session_state['scaler'] = scaler
                        st.success(f"✅ Standardized {len(numerical_columns)} numerical features")
                    
                    # One-hot encode categorical features
                    if encode_categorical and len(categorical_columns) > 0:
                        st.write("**One-hot Encoding Categorical Features:**")
                        encoder = OneHotEncoder(sparse_output=False, drop='first')
                        encoded_features = encoder.fit_transform(X[categorical_columns])
                        
                        # Create dataframe with encoded features
                        encoded_df = pd.DataFrame(
                            encoded_features,
                            columns=encoder.get_feature_names_out(categorical_columns),
                            index=X.index
                        )
                        
                        # Drop original categorical columns and concatenate encoded ones
                        X = pd.concat([X.drop(columns=categorical_columns), encoded_df], axis=1)
                        st.session_state['encoder'] = encoder
                        st.success(f"✅ Encoded {len(categorical_columns)} categorical features → {encoded_df.shape[1]} new features")
                    
                    # Store preprocessed data
                    st.session_state['X'] = X
                    st.session_state['y'] = y
                    st.session_state['preprocessed_data'] = pd.concat([X, pd.Series(y, name=target_column, index=X.index)], axis=1)
                    
                    # Display preprocessed data info
                    st.write("**Preprocessed Data Summary:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Features", X.shape[1])
                    with col2:
                        st.metric("Samples", X.shape[0])
                    with col3:
                        st.metric("Classes", len(np.unique(y)))
                    
                    st.write("**Feature Names after Preprocessing:**")
                    st.write(list(X.columns))
        else:
            st.warning("⚠️ Please select target and feature columns in the sidebar")
    
    with tab4:
        st.header("Model Training")
        
        if 'X' in st.session_state and 'y' in st.session_state:
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Data Split Configuration:**")
                st.write(f"• Training set: {100-int(test_size*100)}%")
                st.write(f"• Test set: {int(test_size*100)}%")
                st.write(f"• Random state: {random_state}")
            
            with col2:
                st.write("**Model Configuration:**")
                st.write(f"• Classifier: Logistic Regression")
                st.write(f"• Strategy: {multiclass_strategy}")
            
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training model..."):
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        st.session_state['X'], 
                        st.session_state['y'],
                        test_size=test_size,
                        random_state=random_state,
                        stratify=st.session_state['y']
                    )
                    
                    # Store split data
                    st.session_state['X_train'] = X_train
                    st.session_state['X_test'] = X_test
                    st.session_state['y_train'] = y_train
                    st.session_state['y_test'] = y_test
                    
                    # Create base classifier - Only Logistic Regression
                    base_classifier = LogisticRegression(max_iter=1000, random_state=random_state)
                    
                    # Apply multi-class strategy
                    if multiclass_strategy == "One-vs-Rest (OvR)":
                        model = LogisticRegression(multi_class='ovr', max_iter=1000, random_state=random_state)
                    elif multiclass_strategy == "One-vs-One (OvO)":
                        model = OneVsOneClassifier(base_classifier)
                    else:  # Auto
                        model = LogisticRegression(multi_class='auto', max_iter=1000, random_state=random_state)
                    
                    # Train model
                    model.fit(X_train, y_train)
                    st.session_state['model'] = model
                    
                    # Make predictions
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    
                    st.session_state['y_pred_train'] = y_pred_train
                    st.session_state['y_pred_test'] = y_pred_test
                    
                    # Calculate metrics
                    train_accuracy = accuracy_score(y_train, y_pred_train)
                    test_accuracy = accuracy_score(y_test, y_pred_test)
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Display training results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Training Accuracy", f"{train_accuracy:.3f}")
                    with col2:
                        st.metric("Test Accuracy", f"{test_accuracy:.3f}")
        else:
            st.warning("⚠️ Please preprocess the data first in the 'Preprocessing' tab")
    
    with tab5:
        st.header("Model Results & Analysis")
        
        if 'model' in st.session_state and st.session_state['model'] is not None:
            
            # Performance Metrics
            st.subheader("📊 Performance Metrics")
            
            y_test = st.session_state['y_test']
            y_pred = st.session_state['y_pred_test']
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # For multi-class, use weighted average
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy:.3f}")
            with col2:
                st.metric("Precision", f"{precision:.3f}")
            with col3:
                st.metric("Recall", f"{recall:.3f}")
            with col4:
                st.metric("F1-Score", f"{f1:.3f}")
            
            # Confusion Matrix
            st.subheader("🔄 Confusion Matrix")
            
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            
            # Add class labels if available
            if 'label_encoder' in st.session_state:
                classes = st.session_state['label_encoder'].classes_
                ax.set_xticklabels(classes, rotation=45, ha='right')
                ax.set_yticklabels(classes, rotation=0)
            
            st.pyplot(fig)
            
            # Classification Report
            st.subheader("📋 Detailed Classification Report")
            
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            
            # Format the dataframe for better display
            report_df = report_df.round(3)
            st.dataframe(report_df, use_container_width=True)
            
            # Feature Importance (if applicable)
            st.subheader("🎯 Feature Importance Analysis")
            
            feature_names = st.session_state['X'].columns.tolist()
            
            # For logistic regression, use coefficient magnitudes
            if hasattr(st.session_state['model'], 'coef_'):
                # Average absolute coefficients across all classes
                importance = np.mean(np.abs(st.session_state['model'].coef_), axis=0)
                
                # Create feature importance dataframe
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                # Plot feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                top_features = importance_df.head(15)  # Show top 15 features
                ax.barh(top_features['Feature'], top_features['Importance'], color='#764ba2')
                ax.set_xlabel('Average Absolute Coefficient')
                ax.set_title('Top 15 Feature Importances (Coefficient Magnitudes)')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Model Export
            st.subheader("💾 Export Model & Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export predictions with all original data
                # Include all original columns from the test set for identification
                predictions_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': y_pred
                })
                
                # Add the original features from test set
                X_test_original = st.session_state['X_test'].copy()
                
                # If we have the original unprocessed data, use it for better readability
                if 'data' in st.session_state and st.session_state['data'] is not None:
                    try:
                        # Get the original indices
                        test_indices = X_test_original.index
                        
                        # Get original data for these indices (all columns except target)
                        original_test_data = st.session_state['data'].loc[test_indices].copy()
                        
                        # Remove target column if it exists
                        if target_column in original_test_data.columns:
                            original_test_data = original_test_data.drop(columns=[target_column])
                        
                        # Combine original data with predictions
                        full_predictions_df = pd.concat([
                            original_test_data.reset_index(drop=True),
                            predictions_df.reset_index(drop=True)
                        ], axis=1)
                    except:
                        # Fallback to processed features if original data retrieval fails
                        full_predictions_df = pd.concat([
                            X_test_original.reset_index(drop=True),
                            predictions_df.reset_index(drop=True)
                        ], axis=1)
                else:
                    # No original data available, use processed features
                    full_predictions_df = pd.concat([
                        X_test_original.reset_index(drop=True),
                        predictions_df.reset_index(drop=True)
                    ], axis=1)
                
                # If label encoder exists, decode the values
                if 'label_encoder' in st.session_state:
                    full_predictions_df['Actual_Label'] = st.session_state['label_encoder'].inverse_transform(y_test)
                    full_predictions_df['Predicted_Label'] = st.session_state['label_encoder'].inverse_transform(y_pred)
                
                csv = full_predictions_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name="predictions_with_full_data.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export classification report
                report_csv = report_df.to_csv()
                st.download_button(
                    label="📥 Download Classification Report",
                    data=report_csv,
                    file_name="classification_report.csv",
                    mime="text/csv"
                )
            
            # Additional Analysis Options
            st.subheader("🔍 Additional Analysis")
            
            with st.expander("Test with Different Parameters"):
                st.write("**Try different test sizes:**")
                
                test_sizes = [0.1, 0.2, 0.3, 0.4]
                results = []
                
                for ts in test_sizes:
                    X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
                        st.session_state['X'],
                        st.session_state['y'],
                        test_size=ts,
                        random_state=42,
                        stratify=st.session_state['y']
                    )
                    
                    # Create a temporary model with same configuration (Logistic Regression only)
                    temp_model = LogisticRegression(max_iter=1000, random_state=42)
                    
                    if multiclass_strategy == "One-vs-Rest (OvR)":
                        temp_model = OneVsRestClassifier(temp_model)
                    elif multiclass_strategy == "One-vs-One (OvO)":
                        temp_model = OneVsOneClassifier(temp_model)
                    
                    temp_model.fit(X_train_temp, y_train_temp)
                    y_pred_temp = temp_model.predict(X_test_temp)
                    acc = accuracy_score(y_test_temp, y_pred_temp)
                    
                    results.append({
                        'Test Size': f"{int(ts*100)}%",
                        'Train Samples': len(X_train_temp),
                        'Test Samples': len(X_test_temp),
                        'Accuracy': f"{acc:.3f}"
                    })
                
                results_df = pd.DataFrame(results)
                st.table(results_df)
        
        else:
            st.info("📌 Please train a model first in the 'Training' tab")
    
    with tab6:
        st.header("🔮 Make Predictions on New Data")
        
        if 'model' in st.session_state and st.session_state['model'] is not None:
            
            st.info("Upload a new dataset or enter data manually to make predictions using your trained model.")
            
            # Get feature columns from training
            if 'feature_columns' in locals():
                expected_features = feature_columns.copy()
            else:
                st.error("❌ Feature information not available. Please train a model first.")
                st.stop()
            
            # Choose input method
            input_method = st.radio(
                "Select input method:",
                ["📁 Upload CSV File", "✏️ Enter Data Manually"],
                horizontal=True
            )
            
            if input_method == "📁 Upload CSV File":
                st.subheader("Upload New Data File")
                
                new_file = st.file_uploader(
                    "Choose a CSV file for prediction",
                    type="csv",
                    key="prediction_file",
                    help="Upload a CSV file with the same features as your training data (excluding target column)"
                )
                
                if new_file is not None:
                    # Load new data
                    new_data = pd.read_csv(new_file)
                    st.success(f"✅ New data loaded! Shape: {new_data.shape}")
                    
                    # Display first few rows
                    st.write("**Preview of new data:**")
                    st.dataframe(new_data.head())
                    
                    # Check if columns match
                    expected_features = feature_columns.copy()
                    missing_cols = set(expected_features) - set(new_data.columns)
                    extra_cols = set(new_data.columns) - set(expected_features)
                    
                    if missing_cols:
                        st.error(f"❌ Missing columns: {missing_cols}")
                        st.stop()
                    
                    if extra_cols:
                        st.warning(f"⚠️ Extra columns will be ignored: {extra_cols}")
                        new_data = new_data[expected_features]
                    
                    if st.button("🎯 Make Predictions", key="batch_predict"):
                        with st.spinner("Processing and making predictions..."):
                            try:
                                # Preprocess the new data
                                processed_new_data = new_data[expected_features].copy()
                                
                                # Handle missing values
                                numerical_cols = processed_new_data.select_dtypes(include=[np.number]).columns
                                for col in numerical_cols:
                                    if processed_new_data[col].isnull().any():
                                        # Use median from training data if available
                                        if 'training_medians' in st.session_state and col in st.session_state['training_medians']:
                                            processed_new_data[col].fillna(st.session_state['training_medians'][col], inplace=True)
                                        else:
                                            processed_new_data[col].fillna(processed_new_data[col].median(), inplace=True)
                                
                                categorical_cols = processed_new_data.select_dtypes(include=['object']).columns
                                for col in categorical_cols:
                                    if processed_new_data[col].isnull().any():
                                        # Use mode from training data if available
                                        if 'training_modes' in st.session_state and col in st.session_state['training_modes']:
                                            processed_new_data[col].fillna(st.session_state['training_modes'][col], inplace=True)
                                        else:
                                            processed_new_data[col].fillna(processed_new_data[col].mode()[0], inplace=True)
                                
                                # Apply the same preprocessing as training data
                                X_new = processed_new_data.copy()
                                
                                # Identify numerical and categorical columns
                                numerical_columns = X_new.select_dtypes(include=[np.number]).columns.tolist()
                                categorical_columns = X_new.select_dtypes(include=['object']).columns.tolist()
                                
                                # Apply scaling if it was used during training
                                if 'scaler' in st.session_state and len(numerical_columns) > 0:
                                    X_new[numerical_columns] = st.session_state['scaler'].transform(X_new[numerical_columns])
                                
                                # Apply encoding if it was used during training
                                if 'encoder' in st.session_state and len(categorical_columns) > 0:
                                    encoded_features = st.session_state['encoder'].transform(X_new[categorical_columns])
                                    encoded_df = pd.DataFrame(
                                        encoded_features,
                                        columns=st.session_state['encoder'].get_feature_names_out(categorical_columns),
                                        index=X_new.index
                                    )
                                    X_new = pd.concat([X_new.drop(columns=categorical_columns), encoded_df], axis=1)
                                
                                # Ensure columns are in the same order as training data
                                X_new = X_new[st.session_state['X'].columns]
                                
                                # Make predictions
                                predictions = st.session_state['model'].predict(X_new)
                                
                                # Get prediction probabilities if available
                                try:
                                    pred_proba = st.session_state['model'].predict_proba(X_new)
                                    has_proba = True
                                except:
                                    has_proba = False
                                
                                # Create results dataframe
                                results_df = new_data.copy()
                                
                                # Add predictions
                                if 'label_encoder' in st.session_state:
                                    results_df['Predicted_Class'] = st.session_state['label_encoder'].inverse_transform(predictions)
                                    results_df['Predicted_Class_Code'] = predictions
                                    
                                    # Add probability columns if available
                                    if has_proba:
                                        for i, class_name in enumerate(st.session_state['label_encoder'].classes_):
                                            results_df[f'Probability_{class_name}'] = pred_proba[:, i]
                                else:
                                    results_df['Predicted_Class'] = predictions
                                    
                                    # Add probability columns if available
                                    if has_proba:
                                        for i in range(pred_proba.shape[1]):
                                            results_df[f'Probability_Class_{i}'] = pred_proba[:, i]
                                
                                st.success("✅ Predictions completed!")
                                
                                # Display results
                                st.subheader("Prediction Results")
                                
                                # Show summary statistics
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Total Predictions", len(results_df))
                                with col2:
                                    if 'label_encoder' in st.session_state:
                                        unique_preds = results_df['Predicted_Class'].value_counts()
                                    else:
                                        unique_preds = pd.Series(predictions).value_counts()
                                    st.metric("Unique Classes Predicted", len(unique_preds))
                                
                                # Display prediction distribution
                                st.write("**Prediction Distribution:**")
                                fig, ax = plt.subplots(figsize=(10, 6))
                                unique_preds.plot(kind='bar', ax=ax, color='#667eea')
                                ax.set_xlabel('Predicted Class')
                                ax.set_ylabel('Count')
                                ax.set_title('Distribution of Predictions')
                                plt.xticks(rotation=45, ha='right')
                                st.pyplot(fig)
                                
                                # Display detailed results
                                st.write("**Detailed Predictions:**")
                                st.dataframe(results_df)
                                
                                # Download predictions
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Predictions (CSV)",
                                    data=csv,
                                    file_name="new_predictions.csv",
                                    mime="text/csv"
                                )
                                
                            except Exception as e:
                                st.error(f"❌ Error making predictions: {str(e)}")
                                st.write("Please ensure the new data has the same structure as the training data.")
            
            else:  # Manual entry
                st.subheader("Enter Data Manually")
                
                # Create input fields for each feature
                st.write("Enter values for each feature:")
                
                # Store input values
                input_data = {}
                
                # Get feature information from training data
                if 'X' in st.session_state:
                    # Original features (before encoding)
                    if 'feature_columns' in locals():
                        original_features = feature_columns
                    else:
                        st.error("Feature information not available")
                        st.stop()
                    
                    # Get sample data for reference
                    if st.session_state['data'] is not None and all(f in st.session_state['data'].columns for f in original_features):
                        original_data_sample = st.session_state['data'][original_features]
                    else:
                        # Create dummy sample if data not available
                        original_data_sample = pd.DataFrame(columns=original_features)
                    
                    # Create columns for better layout
                    col1, col2 = st.columns(2)
                    
                    for i, feature in enumerate(original_features):
                        # Determine input type based on original data type
                        if st.session_state['data'] is not None and feature in st.session_state['data'].columns:
                            # We have the original data, use it for reference
                            if feature in st.session_state['data'].select_dtypes(include=[np.number]).columns:
                                # Numerical input
                                with col1 if i % 2 == 0 else col2:
                                    # Get min/max from training data for reference
                                    min_val = float(st.session_state['data'][feature].min())
                                    max_val = float(st.session_state['data'][feature].max())
                                    mean_val = float(st.session_state['data'][feature].mean())
                                    
                                    input_data[feature] = st.number_input(
                                        f"{feature}",
                                        min_value=min_val - abs(min_val),
                                        max_value=max_val + abs(max_val),
                                        value=mean_val,
                                        help=f"Range in training data: [{min_val:.2f}, {max_val:.2f}]"
                                    )
                            else:
                                # Categorical input
                                with col1 if i % 2 == 0 else col2:
                                    unique_values = st.session_state['data'][feature].unique().tolist()
                                    input_data[feature] = st.selectbox(
                                        f"{feature}",
                                        options=unique_values,
                                        help=f"Select one of the {len(unique_values)} categories"
                                    )
                        else:
                            # No data available, provide generic input
                            with col1 if i % 2 == 0 else col2:
                                # Make a guess based on feature name
                                if any(word in feature.lower() for word in ['category', 'type', 'class', 'gender', 'status']):
                                    input_data[feature] = st.text_input(
                                        f"{feature}",
                                        help="Enter value (likely categorical)"
                                    )
                                else:
                                    input_data[feature] = st.number_input(
                                        f"{feature}",
                                        value=0.0,
                                        help="Enter numerical value"
                                    )
                    
                    if st.button("🎯 Make Prediction", key="single_predict"):
                        with st.spinner("Making prediction..."):
                            try:
                                # Create dataframe from input
                                input_df = pd.DataFrame([input_data])
                                
                                # Apply the same preprocessing as training data
                                X_single = input_df.copy()
                                
                                # Identify numerical and categorical columns
                                numerical_columns = X_single.select_dtypes(include=[np.number]).columns.tolist()
                                categorical_columns = X_single.select_dtypes(include=['object']).columns.tolist()
                                
                                # Apply scaling if it was used during training
                                if 'scaler' in st.session_state and len(numerical_columns) > 0:
                                    X_single[numerical_columns] = st.session_state['scaler'].transform(X_single[numerical_columns])
                                
                                # Apply encoding if it was used during training
                                if 'encoder' in st.session_state and len(categorical_columns) > 0:
                                    encoded_features = st.session_state['encoder'].transform(X_single[categorical_columns])
                                    encoded_df = pd.DataFrame(
                                        encoded_features,
                                        columns=st.session_state['encoder'].get_feature_names_out(categorical_columns),
                                        index=X_single.index
                                    )
                                    X_single = pd.concat([X_single.drop(columns=categorical_columns), encoded_df], axis=1)
                                
                                # Ensure columns are in the same order as training data
                                X_single = X_single[st.session_state['X'].columns]
                                
                                # Make prediction
                                prediction = st.session_state['model'].predict(X_single)[0]
                                
                                # Get prediction probability if available
                                try:
                                    pred_proba = st.session_state['model'].predict_proba(X_single)[0]
                                    has_proba = True
                                except:
                                    has_proba = False
                                
                                # Display result
                                st.success("✅ Prediction completed!")
                                
                                # Show prediction
                                st.subheader("Prediction Result")
                                
                                if 'label_encoder' in st.session_state:
                                    predicted_class = st.session_state['label_encoder'].inverse_transform([prediction])[0]
                                    st.metric("Predicted Class", predicted_class)
                                else:
                                    st.metric("Predicted Class", prediction)
                                
                                # Show probabilities if available
                                if has_proba:
                                    st.write("**Class Probabilities:**")
                                    
                                    if 'label_encoder' in st.session_state:
                                        prob_df = pd.DataFrame({
                                            'Class': st.session_state['label_encoder'].classes_,
                                            'Probability': pred_proba
                                        })
                                    else:
                                        prob_df = pd.DataFrame({
                                            'Class': [f'Class {i}' for i in range(len(pred_proba))],
                                            'Probability': pred_proba
                                        })
                                    
                                    prob_df = prob_df.sort_values('Probability', ascending=False)
                                    
                                    # Display as bar chart
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.barh(prob_df['Class'], prob_df['Probability'], color='#764ba2')
                                    ax.set_xlabel('Probability')
                                    ax.set_title('Prediction Confidence by Class')
                                    ax.set_xlim(0, 1)
                                    for i, (cls, prob) in enumerate(zip(prob_df['Class'], prob_df['Probability'])):
                                        ax.text(prob + 0.01, i, f'{prob:.3f}', va='center')
                                    st.pyplot(fig)
                                    
                                    # Also show as table
                                    st.write("**Probability Details:**")
                                    prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.4f}")
                                    st.table(prob_df)
                                
                                # Show input summary
                                with st.expander("View Input Data"):
                                    st.write("**Your input values:**")
                                    input_summary = pd.DataFrame([input_data])
                                    st.dataframe(input_summary)
                                    
                                    # Add download button for the single prediction
                                    result_data = input_data.copy()
                                    if 'label_encoder' in st.session_state:
                                        result_data['Predicted_Class'] = predicted_class
                                    else:
                                        result_data['Predicted_Class'] = prediction
                                    
                                    if has_proba:
                                        if 'label_encoder' in st.session_state:
                                            for i, cls in enumerate(st.session_state['label_encoder'].classes_):
                                                result_data[f'Prob_{cls}'] = pred_proba[i]
                                        else:
                                            for i in range(len(pred_proba)):
                                                result_data[f'Prob_Class_{i}'] = pred_proba[i]
                                    
                                    result_df = pd.DataFrame([result_data])
                                    csv = result_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Prediction Result",
                                        data=csv,
                                        file_name="single_prediction.csv",
                                        mime="text/csv"
                                    )
                                
                            except Exception as e:
                                st.error(f"❌ Error making prediction: {str(e)}")
                                st.write("Please check your input values and try again.")
                else:
                    st.warning("⚠️ Training data information not available. Please train a model first.")
            
            # Add model information
            st.sidebar.divider()
            st.sidebar.subheader("📊 Model Information")
            if 'model' in st.session_state:
                st.sidebar.write(f"**Classifier:** Logistic Regression")
                
                if 'multiclass_strategy' in locals():
                    st.sidebar.write(f"**Strategy:** {multiclass_strategy}")
                
                if 'label_encoder' in st.session_state:
                    st.sidebar.write(f"**Classes:** {', '.join(map(str, st.session_state['label_encoder'].classes_))}")
                
                if 'X' in st.session_state:
                    st.sidebar.write(f"**Features:** {len(st.session_state['X'].columns)}")
        
        else:
            st.warning("⚠️ No trained model found. Please train a model first in the 'Training' tab.")
            st.info("""
            **Steps to train a model:**
            1. Upload your training data
            2. Select target and feature columns
            3. Preprocess the data
            4. Train the model (using Logistic Regression)
            5. Come back here to make predictions on new data
            """)

else:
    # Landing page when no data is uploaded
    st.markdown("""
    ## Welcome to the Multi-class Classification App! 👋
    
    This application provides a comprehensive solution for multi-class classification on any CSV dataset.
    
    ### 🚀 Getting Started
    
    1. **Upload your CSV file** using the sidebar
    2. **Select your target column** - the variable you want to predict
    3. **Choose feature columns** - the variables to use for prediction
    4. **Configure preprocessing options** - standardization and encoding
    5. **Select model and strategy** - choose classifier and multi-class approach
    6. **Train your model** and analyze results
    
    ### ✨ Features
    
    - **Automatic Feature Detection**: Identifies numerical and categorical columns
    - **Data Preprocessing**: Handles missing values, scaling, and encoding
    - **Multiple Classifiers**: Logistic Regression, SVM, Random Forest
    - **Multi-class Strategies**: One-vs-Rest (OvR) and One-vs-One (OvO)
    - **Comprehensive Analysis**: Confusion matrix, classification report, feature importance
    - **Interactive Visualizations**: Explore data distributions and model performance
    - **Export Results**: Download predictions and reports
    
    ### 📊 Supported Classification Types
    
    - Binary Classification
    - Multi-class Classification
    - Imbalanced Datasets (with stratified splitting)
    
    ### 🎯 Use Cases
    
    - Customer segmentation
    - Disease diagnosis
    - Product categorization
    - Risk assessment
    - Quality control
    - And many more...
    
    ---
    
    **Ready to start?** Upload your dataset using the sidebar! 📁
    """)
    
    # Add sample dataset info
    with st.expander("📝 Sample Dataset Format"):
        st.markdown("""
        Your CSV file should have:
        - **Features**: Columns containing predictor variables (numerical or categorical)
        - **Target**: A column with the classes/categories you want to predict
        
        Example structure:
        """)
        
        sample_data = pd.DataFrame({
            'Feature1': [1.2, 2.3, 3.1, 4.5, 5.0],
            'Feature2': ['A', 'B', 'A', 'C', 'B'],
            'Feature3': [10, 20, 15, 30, 25],
            'Target': ['Class1', 'Class2', 'Class1', 'Class3', 'Class2']
        })
        st.dataframe(sample_data)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | Multi-class Classification Web App</p>
</div>
""", unsafe_allow_html=True)
