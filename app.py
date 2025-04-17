import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import shap
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set up page config
st.set_page_config(page_title="SHAP Analysis Dashboard", layout="wide")

# Setup Korean font support
def setup_korean_font():
    # Check for the malgunsl.ttf font file
    font_path = 'malgunsl.ttf'
    if os.path.exists(font_path):
        # Configure matplotlib to use the Korean font
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_family()
        plt.rcParams['axes.unicode_minus'] = False
        return font_prop
    else:
        st.warning("Korean font file not found. Some Korean characters may not display correctly.")
        return None

# Main function to run the Streamlit app
def main():
    st.title("Seafood Inspection SHAP Analysis Dashboard")
    
    # Set up Korean font
    font_prop = setup_korean_font()
    
    # Display directional impact information
    st.sidebar.markdown("""
    ### Directional Impact
    - **Positive SHAP Values** (🔴): Push predictions toward non-conformity (class 1)
    - **Negative SHAP Values** (🔵): Push predictions toward conformity (class 0)
    """)
    
    # Sidebar for options
    st.sidebar.header("Sample Selection")
    
    # Sample size selection
    sample_size = st.sidebar.slider(
        "Samples per class",
        min_value=20,
        max_value=100,
        value=88,
        step=1
    )
    
    # Number of features to display
    num_features = st.sidebar.slider(
        "Number of features to display",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )
    
    # Button to run analysis
    run_analysis = st.sidebar.button("Run Analysis")
    
    if run_analysis:
        with st.spinner("Loading data..."):
            # Load the dataset
            df = pd.read_csv('seafood_2022_class_and_distributed_method_ratio.csv')
            
            # Create English translation dictionary for feature names (for display only)
            korean_to_english = {
                    '수입화주': 'Importer',
                    '수출국': 'Export Country',
                    '해외제조업소': 'Overseas Manufacturer',
                    '수출업소': 'Exporting Company',
                    '품명_대분류': 'Product Name - Major Category',
                    '품명_중분류': 'Product Name - Middle Category',
                    '품명': 'Product Name',
                    '총수량': 'Total Quantity',
                    '총순중량': 'Total Net Weight',
                    '유통방식': 'Distribution Method',
                    '검사종류_text': 'Inspection Type (Text)',
                    '검사건수': 'Inspection Type (Numbers)',
                    'year': 'Year',
                    'month': 'Month',
                    'week': 'Week',
                    'year_month': 'Year and Month',
                    'year_week': 'Year and Week',
                    'season': 'Season',
                    '수출국_대륙': 'Continent',
                    '수입화주_ratio': 'Importer Ratio',
                    '수출국_ratio': 'Export Country Ratio',
                    '해외제조업소_ratio': 'Overseas Manufacturer Ratio',
                    '수출업소_ratio': 'Exporting Company Ratio',
                    '품명_대분류_ratio': 'Product Name - Major Category Ratio',
                    '품명_중분류_ratio': 'Product Name - Middle Category Ratio',
                    '품명_ratio': 'Product Name Ratio',
                    '수출국_대륙_ratio': 'Continent Ratio',
                    '유통방식_ratio': 'Distribution Method Ratio',
                    '처리결과': 'Result',
            }
            
            # Define categorical columns (using original Korean names)
            category_columns = [
                '수입화주', '수출국', '해외제조업소', '수출업소',
                '품명_대분류', '품명_중분류', '품명',
                '유통방식', '검사종류_text', 'year', 'month', 'week',
                'year_month', 'year_week', 'season', '수출국_대륙'
            ]
            
            # Convert target variable: '부적합' = 1, '적합' = 0
            df['처리결과'] = df['처리결과'].replace('부적합', 1)
            df['처리결과'] = df['처리결과'].replace('적합', 0)
            
            # Keep original column names in DataFrame
            output_column = '처리결과'
            input_columns = df.columns.tolist()
            input_columns.remove(output_column)
            
            # Split data by class
            df_class_0 = df[df[output_column] == 0]
            df_class_1 = df[df[output_column] == 1]
            
            # Select random samples from each class
            df_class_0_sample = df_class_0.sample(sample_size, random_state=42)
            df_class_1_sample = df_class_1.sample(sample_size, random_state=42)
            
            # Combine the samples
            df_balanced = pd.concat([df_class_0_sample, df_class_1_sample])
            
            # Shuffle the combined data
            df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Load label encoders
            try:
                label_encoder = joblib.load('label_encoders.pkl')
                st.sidebar.success("Label encoders loaded successfully.")
            except Exception as e:
                st.sidebar.error(f"Error loading label encoders: {e}")
                # Create fallback label encoders if file doesn't exist
                label_encoder = {}
                for column in category_columns:
                    if column in df_balanced.columns:
                        le = LabelEncoder()
                        le.fit(df_balanced[column])
                        label_encoder[column] = le
                st.sidebar.warning("Created new label encoders as fallback.")
            
            # Apply label encoding
            df_balanced_encoded = df_balanced.copy()
            label_encoder_dict = {}
            
            for column in category_columns:
                if column in df_balanced.columns:
                    try:
                        # Try to use the loaded label encoder
                        label_encoder_dict[column] = label_encoder[column].transform(df_balanced_encoded[column])
                        df_balanced_encoded[column] = label_encoder_dict[column]
                    except Exception as e:
                        st.sidebar.error(f"Error with encoder for {column}: {e}")
                        # Fallback: create a new encoder
                        le = LabelEncoder()
                        df_balanced_encoded[column] = le.fit_transform(df_balanced_encoded[column])
                        label_encoder_dict[column] = df_balanced_encoded[column]
            
            # Split the data into features and target
            X = df_balanced_encoded[input_columns]
            y = df_balanced_encoded[output_column]
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Store original values for display purposes
            original_X_test = df_balanced[input_columns].iloc[X_test.index]
        
        with st.spinner("Loading voting classifier soft model..."):
            # Load only the voting classifier soft model
            model_path = os.path.join('saved_models', 'voting_classifier_soft.pkl')
            try:
                # Use joblib to load the model
                model = joblib.load(model_path)
                st.sidebar.success("Voting classifier soft model loaded successfully.")
            except Exception as e:
                st.sidebar.error(f"Error loading model: {e}")
                
                # Debug information
                if os.path.exists(model_path):
                    st.sidebar.info(f"File exists at {model_path}, size: {os.path.getsize(model_path)} bytes")
                else:
                    st.sidebar.error(f"File not found at {model_path}")
                    # List contents of the directory
                    if os.path.exists('saved_models'):
                        files = os.listdir('saved_models')
                        st.sidebar.write("Files in saved_models directory:")
                        for file in files:
                            st.sidebar.write(f"- {file}")
                return
        
        with st.spinner("Performing SHAP analysis..."):
            # Convert to numpy array for better compatibility with SHAP
            X_train_np = X_train.values
            X_test_np = X_test.values
            feature_names = X_test.columns.tolist()
            
            # Create a background dataset for the explainer - using k-means for efficiency
            X_train_summary = shap.kmeans(X_train_np, 50)
            
            # Define the predict function that returns probabilities for class 1
            def model_predict(X):
                return model.predict_proba(X)[:, 1]
            
            # Create the explainer with the predict function
            explainer = shap.KernelExplainer(model_predict, X_train_summary)
            
            # Calculate SHAP values for a subset of test samples
            n_samples_to_explain = min(10, len(X_test))  # Limit to 10 samples for speed
            test_sample = X_test_np[:n_samples_to_explain]
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(test_sample)
            
            # Translate feature names for display - AFTER SHAP calculation
            translated_feature_names = [korean_to_english.get(name, name) for name in feature_names]
        
        # Display tabs for different samples
        sample_tabs = st.tabs([f"Sample {i+1}" for i in range(n_samples_to_explain)])
        
        # For each sample, create a tab with visualization
        for i, tab in enumerate(sample_tabs):
            with tab:
                # Create two columns for the layout
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Display prediction details
                    # Modified code for class labels to match the image
                    result_type = "Non-conforming" if y_test.iloc[i] == 1 else "Conforming"
                    actual_type = "Non-conforming" if y_test.iloc[i] == 1 else "Conforming"
                    
                    # Calculate predicted probability
                    prob = model.predict_proba(X_test.iloc[i:i+1])[0][1]
                    
                    # Format prediction details
                    st.subheader("Prediction Details")
                    st.markdown(f"**Predicted Probability:** <span style='color:red; background-color:#f0f0f0; padding:3px 8px; border-radius:4px'>{prob:.4f}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Predicted Class:** {result_type}")
                    st.markdown(f"**Actual Class:** {actual_type}")
                    
                    # Status (correct or incorrect prediction)
                    prediction_correct = (y_test.iloc[i] == (prob > 0.5))
                    if prediction_correct:
                        st.markdown("**Status:** <span style='color:white; background-color:green; padding:3px 8px; border-radius:4px'>Correct Prediction</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("**Status:** <span style='color:white; background-color:red; padding:3px 8px; border-radius:4px'>Incorrect Prediction</span>", unsafe_allow_html=True)
                    
                    # Directional impact explanation
                    st.markdown("""
                    **Directional Impact:**
                    - 🔴 Positive values push toward Non-conformity (1)
                    - 🔵 Negative values push toward Conformity (0)
                    """)
                    
                    # Sample details - with both Korean and English names
                    st.subheader("Sample Data:")
                    sample_df = original_X_test.iloc[i].to_frame(name='Value')
                    # Add English translation as second column
                    sample_df['Feature (English)'] = [korean_to_english.get(idx, idx) for idx in sample_df.index]
                    # Reorder columns
                    sample_df = sample_df[['Feature (English)', 'Value']]
                    sample_df = sample_df.head(27)  # Limit display to 10 rows
                    st.dataframe(sample_df)
                    
                    # Feature contribution details
                    st.subheader("Top Feature Contributions:")
                    
                    # Calculate feature contributions
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Korean': feature_names,  # Keep original Korean
                        'Translated': [korean_to_english.get(name, name) for name in feature_names],
                        'Importance': shap_values[i]
                    })
                    
                    # Sort by absolute importance
                    feature_importance['Abs_Importance'] = np.abs(feature_importance['Importance'])
                    feature_importance = feature_importance.sort_values('Abs_Importance', ascending=False)
                    
                    # Display top features with Korean and translated names
                    for _, row in feature_importance.head(num_features).iterrows():
                        orig_feature = row['Korean']
                        display_feature = f"{orig_feature} ({row['Translated']})"
                        importance = row['Importance']
                        original_value = original_X_test.iloc[i][orig_feature]
                        
                        if importance > 0:
                            st.markdown(f"<span style='color:#FF4B4B'>🔴 {display_feature} = {original_value}: +{importance:.4f}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span style='color:#1E88E5'>🔵 {display_feature} = {original_value}: {importance:.4f}</span>", unsafe_allow_html=True)
                
                with col2:
                    # Waterfall plot
                    st.subheader(f"SHAP Waterfall Plot - Sample {i+1}")
                    st.markdown(f"Prediction: {prob:.4f} (Class {1 if prob > 0.5 else 0}), Actual: {y_test.iloc[i]}")
                    
                    # Create waterfall plot with English translation only
                    fig = plt.figure(figsize=(10, 8))

                    # Sort values by magnitude for better visualization
                    idx = np.argsort(-np.abs(shap_values[i]))
                    sorted_values = shap_values[i][idx]
                    sorted_features = [feature_names[j] for j in idx]
                    sorted_translated = [korean_to_english.get(feature_names[j], feature_names[j]) for j in idx]
                    sorted_original_values = [original_X_test.iloc[i][feature_names[j]] for j in idx]

                    # Create feature labels with ENGLISH ONLY and original values
                    feature_labels = [f"{trans} = {value}" for trans, value in 
                                    zip(sorted_translated[:15], sorted_original_values[:15])]
                    
                    
                    # Plot horizontal bar chart of SHAP values
                 
                    y_pos = np.arange(min(15, len(sorted_features)))
                    colors = ['#1E88E5' if v < 0 else '#FF4B4B' for v in sorted_values[:15]]
                    plt.barh(y_pos, sorted_values[:15], align='center', color=colors)
                    plt.yticks(y_pos, feature_labels, fontproperties=font_prop)
                    plt.xlabel('SHAP Value (impact on model output)')
                    plt.title(f"Feature Contributions for Sample {i+1} (Class: {result_type})")
                    plt.tight_layout()
                    st.pyplot(fig)
        
        # # Create a global feature importance tab
        # st.header("Global Feature Importance")
        
        # # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(0)
        # For directional impact, calculate mean (not abs) SHAP values
        mean_shap = shap_values.mean(0)
        
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Korean': feature_names,  # Keep original Korean
            'Translated': [korean_to_english.get(name, name) for name in feature_names],
            'Importance': mean_abs_shap,
            'Direction': mean_shap  # For directional impact
        })
        
        # # Sort by absolute importance
        # feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        
        # # Display as a bar chart with English names
        # fig = plt.figure(figsize=(12, 10))
        
        # # Create a color map based on the direction of impact
        # colors = ['#FF4B4B' if x > 0 else '#1E88E5' for x in feature_importance.head(20)['Direction']]
        
        # # Plot the horizontal bar chart
        # plt.barh(
        #     [f"({e})" for e in zip(
                
        #         feature_importance.head(20)['Translated'])], 
        #     feature_importance.head(20)['Importance'],
        #     color=colors
        # )
        # plt.xlabel('Mean |SHAP Value|')
        # plt.ylabel('Feature', fontproperties=font_prop)
        # plt.title('Feature Importance Based on SHAP Values')
        # plt.tight_layout()
        # st.pyplot(fig)
        
        # Display as a table with Korean, English, and directional impact
        st.subheader("Feature Importance Ranking")
        importance_table = feature_importance.head(20)[['Korean', 'Translated', 'Importance', 'Direction']].copy()
        importance_table.columns = ['Korean Feature', 'English Feature', 'Importance (Abs)', 'Direction']
        
        # Add an indicator column for direction
        importance_table['Impact'] = importance_table['Direction'].apply(
            lambda x: "🔴 Non-conformity" if x > 0 else "🔵 Conformity"
        )
        
        # Format the numeric columns
        importance_table['Importance (Abs)'] = importance_table['Importance (Abs)'].map(lambda x: f"{x:.4f}")
        importance_table['Direction'] = importance_table['Direction'].map(lambda x: f"{x:.4f}")
        
        # Reorder and display the table
        importance_table = importance_table[['Korean Feature', 'English Feature', 'Importance (Abs)', 'Direction', 'Impact']]
        st.table(importance_table)

# Run the app
if __name__ == "__main__":
    main()