import numpy   as np
import pandas  as pd
import seaborn as sns
from matplotlib      import pyplot as plt
from sklearn.metrics import confusion_matrix

# Model evaluation
def plot_CF_aproach1(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    print('Confusion matrix\n\n', cm)
    print('\nTrue Positives(TP) = ', cm[0,0])
    print('\nTrue Negatives(TN) = ', cm[1,1])
    print('\nFalse Positives(FP) = ', cm[0,1])
    print('\nFalse Negatives(FN) = ', cm[1,0])
def plot_CF_aproach2(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    # Tính tổng số mẫu
    total = np.sum(cm)

    # Tạo labels chứa số lượng và %
    labels = np.array([["{0}\n({1:.1f}%)".format(value, (value/total)*100)
                        for value in row] for row in cm])

    # Tạo DataFrame cho confusion matrix
    cm_matrix = pd.DataFrame(
        data    = cm, 
        columns = ['Actual Positive:1' , 'Actual Negative:0'], 
        index   = ['Predict Positive:1', 'Predict Negative:0']
    )

    # Vẽ heatmap với annot là labels
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        data  = cm_matrix, 
        annot = labels, 
        fmt   = '', 
        cmap  = 'YlGnBu'
    )
    plt.title('Confusion Matrix with Percentage')
    plt.show()