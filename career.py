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

#Ансамблевая модель, которая создает несколько деревьев С4.5 и управляет ими
class CustomRandomForest:

    def __init__(self, n_trees=10, max_depth=5, num_bins=3):
        self.n_trees = n_trees #количество деревьев в лесу
        self.max_depth = max_depth #stopping criteria to prevent overfitting
        self.num_bins = num_bins 

    def bootstrap_sample(self, X, y):
        # Creates a random sample of the dataset with replacement (Bagging).
        # So every tree learns from a slightly different perspective of the data.
        n_samples = X.shape[0]

        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        return X[indices], y[indices]
    
    def fit(self, X, y):
        # The training loop. Builds 'n_trees' independent trees.
        self.trees = []
        
        for i in range(self.n_trees):
            # 1. Get a unique mathematical sample for this tree
            X_sample, y_sample = self.bootstrap_sample(X, y)
            
            # 2. Plant and grow the tree using our recursive C4.5 builder
            tree = build_tree(X_sample, y_sample, self.max_depth,current_depth=0, num_bins=self.num_bins)
            self.trees.append(tree)

    def predict(self, X):
        # What is it?: Asks all trees to predict and then holds an election (Majority Voting).
        
        # 1. Get predictions from EVERY tree. 
        # Result shape: (number_of_trees, number_of_samples)
        tree_predictions = np.array([predict_batch(tree, X) for tree in self.trees])
        
        # 2. Swap axes to group predictions by sample instead of by tree
        # New shape: (number_of_samples, number_of_trees)
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        
        final_predictions = []
    
    # 3. Count the votes for each sample
        for votes in tree_predictions:
            classes, counts = np.unique(votes, return_counts=True)
            winner_class = classes[np.argmax(counts)]
            final_predictions.append(winner_class)
            
        return np.array(final_predictions)
    
    # --- Validation Initiation for Stage 4 (Random Forest) ---
print("\n--- Validation Initiation for the Forest ---")

# Dummy Data (6 samples)
X_val = np.array([
    [0.1, 10], [0.9, 15],  # 'wood' profile
    [0.2, 50], [0.8, 55],  # 'concrete' profile
    [0.5, 90], [0.6, 95]   # 'carpet' profile
])
y_val = np.array(['wood', 'wood', 'concrete', 'concrete', 'carpet', 'carpet'])

print("Initializing the Random Forest...")
# We create a forest with 5 trees, max depth 2, and 3 branches per node
my_forest = CustomRandomForest(n_trees=5, max_depth=2, num_bins=3)

print("Training the ensemble (Bagging in progress)...")
my_forest.fit(X_val, y_val)

print("Holding the election (Predicting)...")
forest_predictions = my_forest.predict(X_val)

print(f"Real labels:   {y_val}")
print(f"Forest labels: {forest_predictions}")

# Check if the forest was successfully populated
trees_created = len(my_forest.trees)

if trees_created == 5 and len(forest_predictions) == len(y_val):
    print(f"\n✅ SUCCESS: The Random Forest created {trees_created} non-binary trees and voted successfully!")
else:
    print("\n❌ ERROR: The ensemble failed to build or predict.")

#Test SPlitter / ручное разделение выборки

def train_test_split_manual(X, y, test_ratio=0.2):
    #Перетасовывает набор данных 
    # и разбивает его на обучающие и тестовые наборы.
    n_samples = len(X)

    # Generate a list of shuffled indices
    #Сгенерировать список перетасованных индексов
    shuffled_indices = np.random.permutation(n_samples)

    # Подсчитайте, сколько строк принадлежит тестовому набору
    test_set_size = int(n_samples * test_ratio)

    # Split the indices
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

#Расчет метрик
def calculate_metrics(y_true, y_pred):
    #Вычисляет точность скрупулезность запоминания и F1-Балл с нуля

    classes = np.unique(y_true)

    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    #Accuracy is global (total correct)/(total samples)
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    #For precision, recall and F1, we calculate per class (Macro-Average approach)
    for c in classes:
        #we predicated class C, and it is actually class C
        #Мы указали класс С, и на самом деле это класс С

        TP = np.sum((y_true == c) & (y_pred ==c ))

        # False Positives (FP): We predicted class C, but it is NOT class C
        FP = np.sum((y_true != c) & (y_pred == c))
        
        # False Negatives (FN): We did NOT predict class C, but it actually IS class C
        FN = np.sum((y_true == c) & (y_pred != c))
        
        # Safe math to avoid dividing by zero
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Add to the macro sum
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        
    # Calculate Macro-Averages (divide by the number of unique classes)
    n_classes = len(classes)
    
    return {
        'Accuracy': accuracy,
        'Precision': macro_precision / n_classes,
        'Recall': macro_recall / n_classes,
        'F1-Score': macro_f1 / n_classes
    }

# --- Validation Initiation for Stage 5.1 (Real Data Test) ---
print("\n--- INITIATING FINAL TEST ON KAGGLE DATA ---")

# 1. Extract X (features) and y (target) from our processed Kaggle dataset
# We drop the 'series_id' and 'surface' to keep only the math columns
X_real = df_list.drop(columns=['series_id', 'surface']).values
y_real = df_list['surface'].values

# 2. Split into 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split_manual(X_real, y_real, test_ratio=0.2)

print(f"Data ready. Training on {len(X_train)} series, Testing on {len(X_test)} series.")

# 3. Initialize the Forest
# For real data, we use more trees and more depth.
print("Planting 15 trees...")
real_forest = CustomRandomForest(n_trees=15, max_depth=6, num_bins=4)

print("Training the Random Forest (This might take a minute depending on your CPU)...")
real_forest.fit(X_train, y_train)

print("Predicting on unseen test data...")
y_test_pred = real_forest.predict(X_test)

# 4. Calculate Final Metrics
results = calculate_metrics(y_test, y_test_pred)

print("\n--- 🏆 FINAL METRICS ON KAGGLE DATA 🏆 ---")
print(f"Accuracy:  {results['Accuracy'] * 100:.2f}%")
print(f"Precision: {results['Precision'] * 100:.2f}%")
print(f"Recall:    {results['Recall'] * 100:.2f}%")
print(f"F1-Score:  {results['F1-Score'] * 100:.2f}%")

# Random guessing among 9 surfaces would give ~11% accuracy. 
if results['Accuracy'] > 0.3: 
    print("\n✅ SUCCESS: The model has successfully learned from the sensor data!")
else:
    print("\n❌ ERROR: The model is struggling to learn.")