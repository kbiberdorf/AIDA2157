import pandas as pd
import sqlalchemy as sa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# ==========================================
# 1. INITIALIZE & TRAIN
# ==========================================
SERVER_NAME = r'BLUEBLOCKSS\MSSQLSERVER01'
DATABASE_NAME = 'AIDA_Alberta_Econ'

try:
    conn_str = f"mssql+pyodbc://@{SERVER_NAME}/{DATABASE_NAME}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    engine = sa.create_engine(conn_str)

    # Pulling Gold View from SQL Warehouse
    df = pd.read_sql("SELECT * FROM vw_Pizza_Model_Data", engine).dropna()
    
    # Capture historical boundaries for the Guidance Note
    limits = {
        'inf_min': df['CPI_Inflation'].min(), 'inf_max': df['CPI_Inflation'].max(),
        'earn_min': df['Avg_Earnings'].min(), 'earn_max': df['Avg_Earnings'].max(),
        'grow_min': df['Retail_Growth'].min(), 'grow_max': df['Retail_Growth'].max()
    }

    # Prepare ML Model
    X = df[['CPI_Inflation', 'Avg_Earnings', 'Retail_Growth']]
    y = df['Econ_Status']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KNeighborsClassifier(n_neighbors=5).fit(X_scaled, y)

    print("✅ Supervised Trained ML Model 1: Online and Calibrated.")

# ==========================================
# 2. MODEL INTERACTION LOOP
# ==========================================
    while True:
        print("\n" + "="*65)
        print("🤖 SUPERVISED TRAINED ML MODEL 1: PREDICTOR")
        print("="*65)
        
        # --- THE UPDATED GUIDANCE NOTE ---
        print(f"📌 NOTE: Use inputs based on our Warehouse data points:")
        print(f"   👉 Inflation: {limits['inf_min']:.2f}% to {limits['inf_max']:.2f}%")
        print(f"   👉 Earnings:  ${limits['earn_min']:.2f} to ${limits['earn_max']:.2f}")
        print(f"   👉 Growth:    {limits['grow_min']:.2f}% to {limits['grow_max']:.2f}%")
        print("-" * 65)
        
        try:
            val_in = input("\nEnter Expected Inflation % (or 'exit'): ")
            if val_in.lower() == 'exit': break
            
            inf = float(val_in)
            earn = float(input("Enter Expected Avg Earnings: "))
            grow = float(input("Enter Expected Retail Growth %: "))

            # --- 🛡️ THE HARD-CAPPED GUARDRAILS ---
            if inf > 25 or inf < -10 or grow > 50 or grow < -50 or earn > 10000 or earn <= 0:
                print("\n❌ DATA ERROR: Input values are economically impossible.")
                print("The Supervised Model cannot predict on 'Sci-Fi' or Zero data.")
                continue

            # PREDICTION PHASE
            user_input = pd.DataFrame([[inf, earn, grow]], 
                                     columns=['CPI_Inflation', 'Avg_Earnings', 'Retail_Growth'])
            user_scaled = scaler.transform(user_input)
            prediction = model.predict(user_scaled)[0]

            print(f"\n✅ MODEL PREDICTION: {prediction}")

            # SAVE TO SQL
            save = input("Save this prediction to the tracking table? (y/n): ")
            if save.lower() == 'y':
                result_df = pd.DataFrame({
                    'Timestamp': [pd.Timestamp.now()],
                    'In_Inflation': [inf], 
                    'In_Earnings': [earn], 
                    'In_Growth': [grow],
                    'Model_Prediction': [prediction],
                    'Model_ID': ['Supervised_ML_Model_1']
                })
                result_df.to_sql('Model_Predictions', engine, if_exists='append', index=False)
                print("💾 Saved to [Model_Predictions] table.")

        except ValueError:
            print("❌ ERROR: Please enter numeric values only.")

except Exception as e:
    print(f"❌ CRITICAL SYSTEM ERROR: {e}")

print("\n👋 Model 1 Session Closed.")