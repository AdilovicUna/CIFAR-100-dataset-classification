import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve

def show_results(test_data, test_target, prediction, model, filename):
    # get and print the required data
    confusion_matrix_result = confusion_matrix(test_target, prediction)
    print('Confusion matrix: \n', confusion_matrix_result)
    macro_precision = precision_score(test_target, prediction, average='macro')
    print('Macro precision: ', macro_precision)
    recall_score_result = recall_score(test_target, prediction, average='macro')
    print('Recall score: ', recall_score_result)

    # plot the precision recall curve
    precision, recall, _ = precision_recall_curve(test_target, model.predict_proba(test_data)[:, 1], pos_label=max(model.classes_))
    _, ax = plt.subplots()
    ax.plot(recall, precision, color='blue')

    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')cd 

    plt.savefig('plots/' + filename + ".png")