import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import warnings
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as sklearnPCA
style.use("ggplot")

import time
start_time = time.time()

warnings.filterwarnings('ignore')

# dataset = pd.read_excel (r'C:\Users\eli.phillips\Documents\TCR ML Algorithm\Program Files\TCR Data.xlsx')  # Read in the Excel file
# dataset = pd.read_excel (r'C:\Users\eli.phillips\Documents\TCR ML Algorithm\Program Files\TCR Snapshot.xlsx')  # Read in the Excel file
# dataset = pd.read_excel (r'C:\Users\eli.phillips\Documents\TCR ML Algorithm\Program Files\Original Shock Iso.xlsx')  # Read in the Excel file
dataset = pd.read_excel (r'C:\Users\eli.phillips\Documents\TCR ML Algorithm\Program Files\Chris Shock Iso.xlsx')  # Read in the Excel file
headers = dataset.columns.values  # Pull out all of the headers in the file

X = dataset
Y = dataset

oh = OneHotEncoder(sparse=False) # One Hot Encoder will help to handle non-categorical data
le = LabelEncoder()  # Label Encoder will handle non-categorial data

row_count = len(dataset.index) # Counts the amount of rows in dataset being used
print_classifier = True

micro = [0] # List used to store micro averages across several trials
macro = [0] # List used to store macro averages across several trials
weighted = [0] # List used to store weighted averages across several trials

micro.clear() # Clear micro list
macro.clear() # Clear macro list
weighted.clear() # Clear weighted list

tree_param = 5 # Default value for the max_depth of a Decision Tree
forest_param = 8 # Default value for the n-estimators of a Random Forest
gbm_param = 8 # Default value for the n-estimators of a GBM

# For every column, fill in blank values with "Unknown", then use the Label
# Encoder on the dataset one column at a time
for i in range(len(headers)):
    dataset[headers[i]].fillna('Unknown', inplace=True)
    dataset[headers[i]] = le.fit_transform(dataset[headers[i]].astype(str))

def kmeans(n_clusters):
    global headers

    x = dataset
    x_std = StandardScaler().fit_transform(x)

    sklearn_pca = sklearnPCA(n_components=2)
    Y_sklearn = sklearn_pca.fit_transform(x_std)

    print(Y_sklearn.shape)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(Y_sklearn) # x_std
    x_clustered = kmeans.fit_predict(x_std)
    centroids = np.array(kmeans.cluster_centers_)

    LABEL_COLOR_MAP = {0: 'r', 1: 'g', 2: 'b', 3: 'purple', 4: 'yellow', 5: 'cyan', 6: 'orange', 7: 'black'}

    label_color = [LABEL_COLOR_MAP[l] for l in x_clustered]

    # Plot the scatter diagram
    plt.figure(figsize=(7, 7))

    # plt.scatter(x_std[:, 0], x_std[:, 1], c=label_color, alpha=0.5)  # "Clusters"
    # plt.scatter(x_std[:, 0], x_std[:, 2], c=label_color, alpha=0.5)  # "Clusters"
    plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=label_color, alpha=0.75)  # "Clusters"

    p_Df = pd.DataFrame(data=Y_sklearn, columns=['pc 1', 'pc 2'])

    plt.figure()
    ax = plt.gca()
    p_Df.plot(x='pc 1', y='pc 2', kind='scatter', title='K-Means', c=label_color, legend='reverse', ax=ax) # All points

    centers = [0]
    centers.clear()
    centroid_total = 0
    for j in range(n_clusters):
        for k in range(len(headers)):
            centroid_total = centroid_total + centroids[j, k]
        centroid_avg = centroid_total / 71
        centers.append(centroid_total)
        centroid_total = 0
    print(centroids[:, 0])
    print(centroids[:, 1])

    ax.scatter(centroids[:,0], centroids[:,1], marker="x", color='black', s=100, linewidths=5, zorder=10)

    def LOF():

        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.neighbors import LocalOutlierFactor

        # plt.scatter(X.iloc[:, 0], X.iloc[:, 1], marker="x")
        p_Df.plot(x='pc 1', y='pc 2', kind='scatter', title='K-Means', c=label_color, legend='reverse', xlim=[-5, 13], ax=ax)  # All points
        plt.axis((-10, 10, -10, 10))
        plt.show

        np.random.seed(42)

        # Generate train data
        X_inliers = 0.3 * np.random.randn(100, 2)
        X_inliers = np.r_[X_inliers + 2, X_inliers - 2]

        X_inliers = p_Df

        # Generate some outliers
        X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
        X = np.r_[X_inliers, X_outliers]

        n_outliers = len(X_outliers)
        ground_truth = np.ones(len(X), dtype=int)
        ground_truth[-n_outliers:] = -1

        # fit the model for outlier detection (default)
        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        # use fit_predict to compute the predicted labels of the training samples
        # (when LOF is used for outlier detection, the estimator has no predict,
        # decision_function and score_samples methods).
        y_pred = clf.fit_predict(X)
        n_errors = (y_pred != ground_truth).sum()
        X_scores = clf.negative_outlier_factor_

        plt.title("Local Outlier Factor (LOF)")
        plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points')
        # plot circles with radius proportional to the outlier scores
        radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
        plt.scatter(X[:, 0], X[:, 1], s=1000 * radius, edgecolors='r',
                    facecolors='none', label='Outlier scores')
        plt.axis('tight')
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        plt.xlabel("Prediction Accuracy: %f" % (1-(n_errors/len(X))))
        plt.xlabel("Prediction Failures: %d" % n_errors)
        legend = plt.legend(loc='upper left')
    LOF()

kmeans(4)

plt.show()