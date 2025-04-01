import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Manually creating the confusion matrix based on your values
conf_matrix = np.array([[7732, 1546],
                         [1963, 10305]])

# Plotting the confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Benign', 'Predicted Attack'], 
            yticklabels=['Actual Benign', 'Actual Attack'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()