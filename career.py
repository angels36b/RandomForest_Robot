import pandas as pd
import numpy as np

# Y is the objective variable / Y - это объектная переменная
def load_and_preprocess_data(x_path, y_path):
    print("Loading data...")
    X_raw = pd.read_csv(x_path)
    y_raw = pd.read_csv(y_path)

    # We separate the identification columns from the sensor results.
    # Разделим столбцы идентификаторов от результатов измерений датчиков.
    columnas_sensor = [col for col in X_raw.columns if col not in ['row_id', 'series_id', 'measurement_number']]
    
    print("Calculating mean and standard deviation...")
    # We group the data by the identifier of the series.
    # Группируем данные по идентификатору серии.
    X_group = X_raw.groupby('series_id')[columnas_sensor].agg(['mean', 'std'])
    
    # FIX APPLIED HERE: We assign the list to X_group.columns, not to X_group itself.
    X_group.columns = ['_'.join(col).strip() for col in X_group.columns.values]

    print("Combining characteristics with their labels...")
    data_finally = X_group.merge(y_raw[['series_id', 'surface']], on='series_id', how='inner')
    
    return X_raw, data_finally

# Execution
X_original, df_list = load_and_preprocess_data('career-con-2019/X_train.csv', 'career-con-2019/y_train.csv')

# --- Stage 1 Validation ---
print("\n--- Validation Initiation for Stage 1 ---")

total_original_row = len(X_original)
unique_series = X_original['series_id'].nunique()
process_rows = len(df_list)

print(f"Original rows (128 per series): {total_original_row}")
print(f"Total unique series expected: {unique_series}")
print(f"Rows in our final dataset: {process_rows}")

#calculate the entropy the shannon of entrupy of an array of labels
#y: A numpy array or pandas series containing the class labels
def calculate_entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    #2. Calculate the probabilities for each class
    probabilities = counts / len(y)

    #3. Calculate entropy using the mathematical formula
    #We add 1e-9 because is undefinen and causes a Math Domain Error.
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
    return entropy

print("\n ---Validation Initiation for entropy---")

y_pure = np.array(['wood', 'wood', 'wood','wood'])
entropy_pure = calculate_entropy(y_pure)
print(f"1. Entropy of pure group: {entropy_pure:.4f}(Expected:~0.0)")

y_mixed = np.array(['wood', 'wood', 'concrete', 'wood'])
entropy_mixed = calculate_entropy(y_mixed)
print(f"2. Entropy of 50/50 mixed group:{entropy_mixed:.4f}(Expected:~1.0)")