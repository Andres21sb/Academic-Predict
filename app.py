import pandas as pd
import time
import matplotlib.pyplot as plt


################################################ Funciones basicas ################################################
#metrics for read_csv
start_time = time.time()
# Load the data
#data = pd.read_csv('DATA/Dataset.csv')
data = pd.read_csv('DATA/MiniDataset.csv')
end_time = time.time()

print("Time taken to read the data: ", end_time - start_time)

# print data size
print('data size -> ',data.shape)

#data description
print('Data description in progress...')

#save output of data.describe() to a txt file
with open('Results/describeMini.txt', 'w') as f:
    f.write(data.describe().to_string())
    
print('Data description completed and saved to Results/describeMini.txt')

#plot bar graph for data target and save it
""" data['Target'].value_counts().plot(kind='bar')
plt.title('Target distribution')
plt.savefig('Results/targetDistributionMini.png')
print('Target distribution plot saved to Results/targetDistributionMini.png') """

################################################ Entrenamiento de prediccion ################################################
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import seaborn as sns
import graphviz

#separar data en features y target
X = data.drop(columns=['Target'])
y = data['Target']

#crear clasificador de arbol de decision
clf = DecisionTreeClassifier(criterion='entropy')

#entrenar clasificador
clf.fit(X, y)

# Export the decision tree to a DOT file
export_graphviz(clf, out_file='Results/decision_tree.dot', feature_names=X.columns, class_names=clf.classes_, filled=True)

# Visualize the decision tree
with open('Results/decision_tree.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

#obtener importancia de features
feature_importances = clf.feature_importances_

# Obtener el nombre del feature que tiene la mayor importancia
best_feature_index = feature_importances.argmax()
best_feature_name = X.columns[best_feature_index]

print('Best feature: ', best_feature_name)




################ graficar feature importances para hacer un contraste

""" # Get feature names
feature_names = X.columns

# Create a DataFrame with feature importances
feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sort features by importance
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances_df, palette='viridis')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('Results/featureImportancePlot.png')
print('Feature importances plot saved to Results/featureImportancePlot.png')
plt.show() """