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

#Calculate how mucho entropy is reduced after a split

def calculate_information_gain(parent_y, children_y_list):
  
   
    #calculate the initial chaos (Entropy of the parent)
    parent_entropy = calculate_entropy(parent_y)
   
    #calculate the weighted chaos of the branches
    total_samples = len(parent_y)
    weighted_children_entropy = 0.0

    #Why a loop? because in C4.5 (non-binary), a split can produce 2, 3 or more branches.
    for child_y in children_y_list:
    
        if len(child_y) == 0:
            continue

        weight = len(child_y) / total_samples
        weighted_children_entropy += weight * calculate_entropy(child_y)

#Gain is the initial chaos minus the remaining chaos

    information_gain = parent_entropy - weighted_children_entropy

    return information_gain

#Validation
print("\n --- Validation Initation for Information gain ---")

parent_node = np.array(['wood', 'wood', 'concrete', 'concrete'])

perfect_split = [
    np.array(['wood', 'wood']),
    np.array(['concrete', 'concrete'])
]
gain_perfect = calculate_information_gain(parent_node, perfect_split)
print(f"1. Gain of a Perfect split:{gain_perfect:.4f} (Expected:~1.0)")


## Test 2: A TERRIBLE SPLIT
# We split the parent, but both branches are still perfectly mixed
terrible_split = [
    np.array(['wood', 'concrete']), 
    np.array(['wood', 'concrete'])
]
gain_terrible = calculate_information_gain(parent_node, terrible_split)
print(f"2. Gain of a TERRIBLE split: {gain_terrible:.4f} (Expected: ~0.0)")

parent_3 = np.array(['wood', 'wood', 'concrete', 'carpet', 'carpet', 'carpet'])
multi_split = [

    np.array(['wood', 'wood']),
    np.array(['concrete']),
    np.array(['carpet', 'carpet', 'carpet'])
]
gain_multi = calculate_information_gain(parent_3, multi_split)
print(f"3. Gain of a PERFECT NON-BINARY split: {gain_multi:.4f} (Expected: > 1.0)")

if gain_perfect > 0.9 and gain_terrible < 0.1 and gain_multi > 1.0:
    print("✅ SUCCESS: The Information Gain engine correctly identifies good and bad non-binary splits.")
else:
    print("❌ ERROR: The calculation is incorrect.")

#function to force non-binary splits / Функция для небинарного разбения 
def split_data_multiway(X, y, column_idx, num_bins=3):
    #Divide a continuos column into 'num_bins' branches( low, medium, high)
    column_data = X[:, column_idx]
    #CACULATE percentiles to find the cutting points
    # For 3 bins, it creates cuts at 33.3% and 66.6%
    edges = np.percentile(column_data, np.linspace(0, 100, num_bins + 1))
   
    # We use np.digitize to assign each row to branch 0, 1, or 2
    # edges[1:-1] ignores the 0% and 100% to only use the internal cuts
    branch_indices = np.digitize(column_data, edges[1:-1])
    children_X = []
    children_y = []

    # Separate the data into the new branches
    for i in range(num_bins):
        mask = (branch_indices == i)
        children_X.append(X[mask])
        children_y.append(y[mask])
        
    return children_X, children_y

#2. Function to find the best feature / Функция поиска лучшего признака

def find_best_split(X, y, feature_indices, num_bins=3):
    #Tests all collumns and returns the one that gives the highest Information Gain
    best_gain = -1
    best_feature = None
    best_children_X = []
    best_children_y = []

    for feature_idx in feature_indices:
        # We try splitting the current feature into multiple branches
        children_X, children_y = split_data_multiway(X, y, feature_idx, num_bins)
        
        # We use our mathematical engine from Stage 2
        gain = calculate_information_gain(y, children_y)
        
        # If this feature organizes the data better, we save it
        if gain > best_gain:
            best_gain = gain
            best_feature = feature_idx
            best_children_X = children_X
            best_children_y = children_y
            
    return best_feature, best_gain, best_children_X, best_children_y

# --- Validation Initiation for Stage 3.1 (Non-Binary Splitter) ---
print("\n--- Validation Initiation for Splitter ---")

# Dummy Data (Rows: 6, Columns: 2)
# Feature 0: Chaos/Noise.  Feature 1: Perfect Predictor.
X_val = np.array([
    [0.1, 10], [0.9, 15],  # Low feature 1 -> Wood
    [0.2, 50], [0.8, 55],  # Med feature 1 -> Concrete
    [0.5, 90], [0.6, 95]   # High feature 1 -> Carpet
])
y_val = np.array(['wood', 'wood', 'concrete', 'concrete', 'carpet', 'carpet'])

# We tell it to test both column 0 and column 1
features_to_test = [0, 1]

best_feat, gain, child_X, child_y = find_best_split(X_val, y_val, features_to_test, num_bins=3)

print(f"1. Best column chosen: {best_feat} (Expected: 1)")
print(f"2. Information Gain: {gain:.4f} (Expected: > 1.0, meaning perfect split)")
print(f"3. Number of branches created: {len(child_y)} (Expected: 3)")

if best_feat == 1 and len(child_y) == 3:
    print("✅ SUCCESS: The algorithm successfully bypassed binary restrictions and found the perfect multi-way split!")
else:
    print("❌ ERROR: The splitter failed.")

##СОЗДАНИЕ ДЕРЕВА
#структура данных

class TreeNode:
    #is object that stores the rules of a specific split in the tree
    #It allows us to easily navigate from the root down to the leaves during prediction.
    #Является ли объект частью правил конкретного разделения в дереве

    def __init__(self, feature_idx=None, edges=None, children=None, value=None):
        self.feature_idx = feature_idx  #the Column index used for the split
        self.edges = edges              #Процентные интервалы  которые следует запомнить для дальнейшего анализа данных/ The percentile cuts (bins) to remember for future data
        self.children = children
        self.value = value

#2. The recursive builder // Рекурсивный строитель
#рекурсивная функция, которая растёт дерево уровень за уровнем
def build_tree(X, y, max_depth, current_depth =0, num_bins =3):
    unique_classes, counts = np.unique(y, return_counts=True)

    #1. Stopping criteria
    # Если мы достигли максимальной глубины ИЛИ узел полностью чистый // If we reached max depth OR the node is completely pure
    
    if current_depth >= max_depth or len(unique_classes) == 1:
        #мы останавливаем рост  и возвращаем Литс с самым частым классом
        majority_class = unique_classes[np.argmax(counts)]
        return TreeNode(value=majority_class)

    #2. Find the best column to split
    # Найдите лучший признак для разбиения
    features_to_test = range(X.shape[1])
    beast_feat, best_gain, child_X, child_y = find_best_split(X, y, features_to_test, num_bins)

    #3. Если мы не можем найти ни одного разбиения, которое улучшит качество информации (Gain <= 0), то
    if best_gain <= 0:
        majority_class = unique_classes[np.argmax(counts)]
        return TreeNode(value=majority_class)
    #4. Запомните границы (процентильные срезы) для этого конкретного признака!
    #Это нужно, чтобы дерево знало, как в дальнейшем направлять новые, невидимые ранее данные.»
    edges = np.percentile(X[:, best_feat], np.linspace(0,100,num_bins + 1))

    #5. Отращивайте ветви // Grow the branches
    children_nodes = []
    for i in range(len(child_X)):
        #Если ветвь осталась без данных, создайте лист для подстраховки
        if len(child_y[i]) == 0:
            majority_class = unique_classes[np.argmax(counts)]
            children_nodes.append(TreeNode(value=majority_class))
        else:
            child_node = build_tree(child_X[i], child_y[i], max_depth, current_depth +1, num_bins)
            children_nodes.append(child_node)
    
    #6. верните внутренний узел, содержащий его дочерние узлы и правила.
    return TreeNode(feature_idx=best_feat, edges=edges, children=children_nodes)

# --- Validation Initiation for Stage 3.2 (Tree Builder) ---

print("\n--- Validation Initiation for Recursive Tree ---")

# We reuse the dummy data from the previous step
X_val = np.array([
    [0.1, 10], [0.9, 15],  # Should be 'wood'
    [0.2, 50], [0.8, 55],  # Should be 'concrete'
    [0.5, 90], [0.6, 95]   # Should be 'carpet'
])
y_val = np.array(['wood', 'wood', 'concrete', 'concrete', 'carpet', 'carpet'])

# Grow the tree!
print("Planting the seed and growing the tree...")
my_first_tree = build_tree(X_val, y_val, max_depth=2, num_bins=3)

# Let's inspect the Root Node
print("\nTree Inspection:")
print(f"- Is it a leaf? {'Yes' if my_first_tree.value else 'No, it has branches'}")
if not my_first_tree.value:
    print(f"- Best feature chosen for Root: Column {my_first_tree.feature_idx}")
    print(f"- Number of non-binary branches: {len(my_first_tree.children)}")
    
    # Check the first child
    first_child = my_first_tree.children[0]
    print(f"- First branch prediction (Leaf value): {first_child.value}")

if my_first_tree.feature_idx == 1 and len(my_first_tree.children) == 3 and my_first_tree.children[0].value == 'wood':
    print("\n✅ SUCCESS: The tree successfully grew recursively, respected the non-binary rule, and stopped correctly!")
else:
    print("\n❌ ERROR: The tree did not build correctly.")

def predict_single(node, row):
    #обходит дерево в поисках одной строки данных, пока не дойдёт до конечного узла
        if node.value is not None:
            return node.value
        
        feature_val = row[node.feature_idx]

        # We use the saved 'edges' to find which branch we should take.
          # np.digitize tells us exactly the index of the child branch.
        # Note: [1:-1] ignores the 0% and 100% boundaries, just like we did during training.
        branch_index = np.digitize([feature_val], node.edges[1:-1])[0]
    
        # Safety mechanism: ensure the index doesn't go out of bounds 
        # (just in case new data is extremely larger or smaller than training data)
        branch_index = np.clip(branch_index, 0, len(node.children) - 1)
    
        # RECURSION: Go deeper into the tree using the chosen child node
        return predict_single(node.children[branch_index], row)
    
    # 2. Function to predict a whole dataset / Функция предсказания для выборки
def predict_batch(tree, X):
    # What is it?: A helper function that applies predict_single to all rows in our matrix X.
    # Why?: We need this to quickly calculate Accuracy later.
    
    predictions = []
    for row in X:
        pred = predict_single(tree, row)
        predictions.append(pred)
        
    return np.array(predictions)
        
    # --- Validation Initiation for Stage 3.3 (Predictor) ---
print("\n--- Validation Initiation for Tree Prediction ---")

# We use the dummy data and the 'my_first_tree' we just built in the previous step
X_val = np.array([
    [0.1, 10], [0.9, 15],  # Real classes: 'wood'
    [0.2, 50], [0.8, 55],  # Real classes: 'concrete'
    [0.5, 90], [0.6, 95]   # Real classes: 'carpet'
])
y_real = np.array(['wood', 'wood', 'concrete', 'concrete', 'carpet', 'carpet'])

print("Executing predictions...")
# We ask our tree to predict the whole dataset
y_pred = predict_batch(my_first_tree, X_val)

print(f"\nReal labels:      {y_real}")
print(f"Predicted labels: {y_pred}")

# Calculate a quick Accuracy
correct_predictions = np.sum(y_real == y_pred)
total_samples = len(y_real)
accuracy = correct_predictions / total_samples

print(f"\nTree Accuracy on Training Data: {accuracy * 100:.2f}%")

if accuracy == 1.0:
    print("✅ SUCCESS: The tree traversed perfectly and predicted all classes correctly!")
else:
    print("❌ ERROR: The prediction logic failed or the tree didn't memorize the pure classes.")