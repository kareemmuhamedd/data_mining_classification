from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
# Apply KNN classifier


def apply_knn_classifier(K, X_train, X_test, y_train):
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train, y_train) # training 
    return knn.predict(X_test) # testing

# Apply Naive Bayes classifier


def apply_naive_bayes_classifier(X_train, X_test, y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    return gnb.predict(X_test)

# Apply decision tree classifier


def apply_decision_tree_classifier(X_train, X_test, y_train):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    return dt.predict(X_test), dt

# Apply randomForest classifier


# def apply_random_forest_classifier(X_train, X_test, y_train):
#     rm = RandomForestClassifier(
#         n_estimators=10, max_depth=25, criterion="gini", min_samples_split=10)
#     rm.fit(X_train, y_train)
#     rm_prd = rm.predict(X_test)
#     return rm, rm_prd


def best_classifier_function(X_train, X_test, y_train, y_test):
    # KNN
    K = round(np.sqrt(len(X_train)))
    y_pred_knn = apply_knn_classifier(K, X_train, X_test, y_train)
    acc_knn = metrics.accuracy_score(y_test, y_pred_knn) # how many correct data we get 

    # Naive Bayes
    y_pred_nb = apply_naive_bayes_classifier(X_train, X_test, y_train)
    acc_nb = metrics.accuracy_score(y_test, y_pred_nb)

    # Decision Tree
    y_pred_dt, _ = apply_decision_tree_classifier(X_train, X_test, y_train)
    acc_dt = metrics.accuracy_score(y_test, y_pred_dt)

    # Compare and return the name of the best classifier
    classifiers = {'KNN': acc_knn,
                   'Naive Bayes': acc_nb,
                   'Decision Tree': acc_dt,
                  }
    best_classifier = max(classifiers, key=classifiers.get)

    print("Accuracy of KNN: {:.2%}".format(acc_knn))
    print("Accuracy of Naive Bayes: {:.2%}".format(acc_nb))
    print("Accuracy of Decision Tree: {:.2%}".format(acc_dt))

    print("\nBest Classifier: {}".format(best_classifier))

    return best_classifier
