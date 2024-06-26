import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os, json
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score







out_path = os.path.join(os.getcwd(), "outs")
os.makedirs(out_path, exist_ok=True)


X_train_final = pd.read_csv(os.path.join(out_path, "process_train_x.csv"))
X_test_final = pd.read_csv(os.path.join(out_path, "process_test_x.csv"))

y_train = pd.read_csv(os.path.join(out_path, "process_train_y.csv"))
y_test = pd.read_csv(os.path.join(out_path, "process_test_y.csv"))

y_train = y_train.iloc[:, 0]
y_test = y_test.iloc[:, 0]
# Count number of occurrences of each value in array fron zero to large...


# to reverse ratio to add it as weights for model
val_count = 1 - (np.bincount(y_train) / len(y_train))
val_count = val_count / np.sum(val_count) # To nurmalize


dict_weight = {}

for i in range(y_train.nunique()):
    dict_weight[i] = val_count[i]
dict_weight

smote = SMOTE(sampling_strategy=0.8)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train_final, y_train)


combined_dict = {}
def train_model(X_train, y_train, plot_name="", class_weight=None):
    global clf_name
    clf = DecisionTreeClassifier(max_depth=15, splitter="best", criterion="gini", class_weight=class_weight)
    clf.fit(X_train, y_train)

    y_test_predict = clf.predict(X_test_final)

    f1_test = f1_score(y_test, y_test_predict)
    acc_test = accuracy_score(y_test, y_test_predict)

    clf_name = clf.__class__.__name__

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  
    sns.heatmap(confusion_matrix(y_test, y_test_predict), annot=True, fmt=".2f", cmap="Blues", cbar=False)
    
    plt.title(f'{plot_name}')
    plt.xticks(ticks=np.arange(2) + 0.5, labels=[False, True])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=[False, True])
    
    plt.savefig(f"{plot_name}.png", bbox_inches="tight", dpi=300)
    plt.close()

    new_result = {f"F1-Test-{plot_name}": f1_test, f"Accuracy-Test-{plot_name}": acc_test}
    global combined_dict
    combined_dict.update(new_result)
   

    return True

train_model(X_train=X_train_final, y_train=y_train, plot_name="without-imbalance", class_weight=None)
train_model(X_train=X_train_final, y_train=y_train, plot_name="with-class-weight", class_weight=dict_weight)
train_model(X_train=X_train_resampled, y_train=y_train_resampled, plot_name="with-SMOT", class_weight=None)

with open("metrics.json", "w") as f:
    json.dump(combined_dict, f)



paths = ["without-imbalance.png", "with-class-weight.png", "with-SMOT.png"]

plt.figure(figsize=(10, 30))


for i, path in enumerate(paths, start=1):
    img = Image.open(path)
    plt.subplot(1, len(path), i)
    plt.axis("off")
    plt.imshow(img)

plt.title(clf_name, fontsize=8)

plt.savefig("Confusion_Matrix.png", bbox_inches="tight", dpi=300)

for path in paths:
    os.remove(path)