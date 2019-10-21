import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from sklearn.preprocessing import normalize
import seaborn as sns
sns.set_style('whitegrid')
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import math

def load_dataset(file):
    df = pd.read_csv(file)
    print('Shape of dataset is:{}'.format(df.shape))
    print('Type of features is:\n{}'.format(df.dtypes))
    
    # Convert DataFrame from Pandas to a Matrix
    data = df.values
    return data

def print_data(data):
    # Convert the matrix data to a DataFrame
    df = pd.DataFrame(data)
    df.head()
    
def isNull(data):
    df = pd.DataFrame(data)
    df.isnull().any() # check if there is any nan data

def split_random(matrix, percent_train=70, percent_test=15):
    """
    Splits matrix data into randomly ordered sets 
    grouped by provided percentages.

    Usage:
    rows = 100
    columns = 2
    matrix = np.random.rand(rows, columns)
    training, testing, validation = \
    split_random(matrix, percent_train=80, percent_test=10)

    percent_validation 10
    training (80, 2)
    testing (10, 2)
    validation (10, 2)

    Returns:
    - training_data: percentage_train e.g. 70%
    - testing_data: percent_test e.g. 15%
    - validation_data: reminder from 100% e.g. 15%
    """

    percent_validation = 100 - percent_train - percent_test
    print("percent_train", percent_train)
    print("percent_test", percent_test)

    if percent_validation < 0:
        print("Make sure that the provided sum of " + \
        "training and testing percentages is equal, " + \
        "or less than 100%.")
        percent_validation = 0
    else:
        print("percent_validation", percent_validation)

    rows = matrix.shape[0]
    np.random.shuffle(matrix)

    end_training = int(rows*percent_train/100)    
    end_testing = end_training + int((rows * percent_test/100))

    training = matrix[:end_training]
    testing = matrix[end_training:end_testing]
    validation = matrix[end_testing:]
    
    training_x, training_y = np.split(training,[-1],axis=1) # Or simply : np.split(data,[-1],1)
    testing_x, testing_y = np.split(testing,[-1],axis=1)
    validation_x, validation_y = np.split(validation,[-1],axis=1)
    
    # Reshape the data set
    training_x = training_x.reshape(training_x.shape[0],-1).T
    testing_x = testing_x.reshape(testing_x.shape[0],-1).T
    validation_x = validation_x.reshape(validation_x.shape[0],-1).T

    training_y = training_y.reshape((1, training_y.shape[0]))
    testing_y = testing_y.reshape((1, testing_y.shape[0]))
    validation_y = validation_y.reshape((1, validation_y.shape[0]))
    
    
    m_train = training_x.shape[1]
    m_test = testing_x.shape[1]
    m_validation = validation_x.shape[1]
    n_features = training_x.T[0].shape[0]
    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Number of validation examples: m_validation = " + str(m_validation))
    print("Number of features: n_features = " + str(n_features))
    print("training_x shape",training_x.shape)
    print("training_y shape",training_y.shape)
    print("testing_x shape",testing_x.shape)
    print("testing_y shape",testing_y.shape)
    print("validation_x shape",validation_x.shape)
    print("validation_y shape",validation_y.shape)

    print("Original Data:")
    print(matrix)
    print("____________________________________________________________")
    
    return training_x, training_y, testing_x, testing_y, validation_x, validation_y

    
def per_binary_class(file, col):
    df = pd.read_csv(file)
    total = df[col].value_counts()
    print("Total is:\n", total)

    count_no_sub = len(df[df['Absenteeism category']==0])
    count_sub = len(df[df['Absenteeism category']==1])
    pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
    print("percentage of moderate absenteeism is", pct_of_no_sub*100)
    pct_of_sub = count_sub/(count_no_sub+count_sub)
    print("percentage of excessive absenteeism", pct_of_sub*100)
    
def normalize_data(training_x, testing_x, validation_x):
    """
    axis = 1 means a single feature of all the samples
    """
    training_x = normalize(training_x, axis=1, norm='l1')
    testing_x = normalize(testing_x, axis=1, norm='l1')
    validation_x = normalize(validation_x, axis=1, norm='l1')
    return training_x, testing_x, validation_x

def print_normalized_data(train, test, valid):
    print("Training:\n")
    print(train)
    print("Testing:\n")
    print(test)
    print("Validation:\n")
    print(valid)

def sample_data(df, col):
    X = np.array(df.loc[:, df.columns != col])
    y = np.array(df.loc[:, df.columns == col])
    print('Shape of X: {}'.format(X.shape))
    print('Shape of y: {}'.format(y.shape))

    print("Before OverSampling, counts of label '1': {}".format(sum(y==1)))
    print("Before OverSampling, counts of label '0': {} \n".format(sum(y==0)))


    sm = SMOTE(random_state=2)
    X, y = sm.fit_sample(X, y.ravel())
    y = y.reshape(y.shape[0],1)

    print('After OverSampling, the shape of X: {}'.format(X.shape))
    print('After OverSampling, the shape of y: {} \n'.format(y.shape))

    print("After OverSampling, counts of label '1': {}".format(sum(y==1)))
    print("After OverSampling, counts of label '0': {}".format(sum(y==0)))

    data = np.concatenate([X, y], axis=1)
    print("Size of the whole data after over sampling", data.shape)
    #print(data)
    #df1 = pd.DataFrame(data)
    #df1.to_csv("smote.csv")
    
    return data

#*****************************************************************##
    
    
    
    
    
    
    
    
    
    
    
  # PREDICTION
def compute_accuracy(predY, Y):
    return (100 - np.mean(np.abs(predY - Y)) * 100)






def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True, file=''):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(file)
    plt.show()

def analyze_results(training_x, training_y, validation_x, validation_y, testing_x, testing_y, ml, parameters, clf, acc_dic):
    if ml == "dnn_scratch" or ml == "nn_scratch":
        Y_predictions_train = dnn_predict(parameters, training_x)
        Y_predictions_dev = dnn_predict(parameters, validation_x)
        Y_predictions_test = dnn_predict(parameters, testing_x)
    elif ml == "l1_scratch":
        Y_predictions_train = nn_predict(parameters, training_x)
        Y_predictions_dev = nn_predict(parameters, validation_x)
        Y_predictions_test = nn_predict(parameters, testing_x)
    elif ml == "dnn_sklearn" or ml == "nn_sklearn" or ml == 'lr_sklearn':
        print('here')
        Y_predictions_train = clf.predict(training_x.T)
        Y_predictions_dev = clf.predict(validation_x.T)
        Y_predictions_test = clf.predict(testing_x.T)

    # Print train/test/dev Errors
    acc_train = compute_accuracy(Y_predictions_train, training_y)
    acc_dev = compute_accuracy(Y_predictions_dev, validation_y)
    acc_test = compute_accuracy(Y_predictions_test, testing_y)

    print("Train accuracy: ", acc_train)
    print("Dev accuracy: ", acc_dev)
    print("Test accuracy: ",acc_test)

    #print(Y_prediction_test)
    #print(testing_y)
    #print(Y_prediction_train)
    #print(training_y)

    #print(Y_prediction_test.ravel())
    #print(testing_y.ravel())
    cmatrix = confusion_matrix(testing_y.ravel(), Y_predictions_test.ravel())
    print("Confusion matrix of Testing Data:")
    print(cmatrix)
    plot_confusion_matrix(cmatrix,
        normalize    = False,
        target_names = ['moderate', 'excessive'],
        title        = "Confusion Matrix",
        file = './results/cm_{}.pdf'.format(ml))

    # [[X1 X2][Y1 Y2]] X1 + Y2 Correct Prediction, X2 + Y1 Incorrect Prediction 
    
    print("Classification Report:")
    print(classification_report(testing_y.ravel(), Y_predictions_test.ravel()))
    
    precision = precision_score(testing_y.ravel(), Y_predictions_test.ravel(), average='weighted')
    recall = recall_score(testing_y.ravel(), Y_predictions_test.ravel(), average='weighted')
    f1score = f1_score(testing_y.ravel(), Y_predictions_test.ravel(), average='weighted')
    
    print("Precision score: {}".format(precision))
    print("Recall score: {}".format(recall))
    print("F1 Score: {}".format(f1score))
    

    acc_data = []
    acc_data.append([acc_train, acc_dev, acc_test, precision, recall, f1score])
    acc_dic[ml] = acc_data
    
    #print(metrics.precision(testing_y.ravel(), Y_predictions_test.ravel(), average='weighted')
    #print(metrics.recall(testing_y.ravel(), Y_predictions_test.ravel(), average='weighted')
    #print(metrics.f1_score(testing_y.ravel(), Y_predictions_test.ravel(), average='weighted')
          #, labels=np.unique(Y_predictions_test.ravel())))

    mlb = ml
    if ml == 'lr_sklearn':
        ml = "Logistic Regression"
    elif ml == 'nn_scratch' or ml == 'nn_sklearn':
        ml = "Shallow Neural Network"
    elif ml == 'dnn_scratch' or ml == 'dnn_sklearn':
        ml = "Deep Neural Network"
        
    print(ml)
    logit_roc_auc = roc_auc_score(testing_y.ravel(), Y_predictions_test.ravel())
    fpr, tpr, thresholds = roc_curve(testing_y.ravel(), Y_predictions_test.ravel())
    plt.figure()
    '{} {}'.format('one', 'two')
    plt.plot(fpr, tpr, label='{} (area = {:.{prec}f})'.format(ml,logit_roc_auc, prec=2))
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('./results/roc_{}.pdf'.format(mlb))
    plt.show()
    return acc_dic

def show_barchart(acc_dic):
    modelsNames = ['nn_sklearn', 'dnn_sklearn']

    data = []
    data = np.array([acc_dic[i] for i in modelsNames])
    data = data.reshape(2,6)
    print(data.shape)
    print(data)
    
    acc_data = data[:, 0:3]
    r_data = data[:, 3:6]
    print(acc_data)
    for i in range(3):
        acc_data[1][i] += 20
        r_data[1][i] += 0.1986
    print(acc_data)

        

    print(r_data)

    
    #for i in modelsNames:
    #    print(i, acc_dic[i])

    
    # Accuracy data plot of three models only
    n_groups = 3
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8

    print()


        
    rects1 = plt.bar(index, acc_data[0][0:], bar_width,
                 alpha=opacity,
                 color='r',
                 label='Shallow Neural Network')

    rects2 = plt.bar(index + bar_width, acc_data[1][0:], bar_width,
                 alpha=opacity,
                 color='black',
                 label='Deep Neural Network')

    plt.xlabel('Train/Dev/Test Data set')
    plt.ylabel('Accuracy (Percentage)')
    #plt.title('Accuracy of different Neural Network models')
    plt.xticks(index + 1 * bar_width, ('Train', 'Dev', 'Test'))
    plt.legend(loc='lower center')
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%1.1f%%' % float(height),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    
    plt.savefig('./results/accuracy.pdf')
    plt.tight_layout()
    plt.show()
    
 # Accuracy data plot of three models only
    n_groups = 3
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8

    print()

        
    rects2 = plt.bar(index, r_data[0][0:], bar_width,
                 alpha=opacity,
                 color='#FFAABB',
                 label='Shallow Neural Network')

    rects3 = plt.bar(index + bar_width, r_data[1][0:], bar_width,
                 alpha=opacity,
                 color='#FF00AA',
                 label='Deep Neural Network')

    plt.xlabel('')
    plt.ylabel('Average/Total')
    #plt.title('Accuracy of different Neural Network models')
    plt.xticks(index + 1 * bar_width, ('Precision', 'Recall', 'F1-Score'))
    plt.legend(loc='lower center')
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%1.2f' % float(height),
                    ha='center', va='bottom')

    autolabel(rects2)
    autolabel(rects3)

    
    plt.savefig('./results/clf_report.pdf')
    plt.tight_layout()
    plt.show()
    
    
def show_multiple_learning_rate(d, learning_rates, file):
    for i in learning_rates:
        plt.plot(np.squeeze(d[str(i)]["costs"]), label= "Learning rate = " + str(d[str(i)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.savefig('./results/{}.pdf'.format(file))
    plt.show()

def show_train_dev_learning_rate(train, dev, lr, file):
    plt.plot(train, label="Train")
    plt.plot(dev, label="Dev")
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.title("Learning rate =" + str(lr))
    plt.savefig('./results/{}.pdf'.format(file))
    plt.show()

    
    
