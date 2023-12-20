import numpy as np
from sklearn.metrics import classification_report
from data_classification_part.classifier_functions import *
from utils import *
from sklearn import tree
import graphviz


import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
# from classifier_functions import *
from utils import *
from sklearn import tree
import graphviz


# def draw_the_most_important_feature(best_one, X_train,df,colors='viridis'):
#     # Assuming X_train and dt are available in the global scope
#     feature_names = X_train.columns
#     feature_imports = df.feature_importances_

#     # Create a DataFrame with feature names and importances, then select the top 10
#     most_imp_features = pd.DataFrame([f for f in zip(feature_names, feature_imports)], columns=[
#                                      "Feature", "Importance"]).nlargest(10, "Importance")
#     most_imp_features.sort_values(by="Importance", inplace=True)

#     # Use a colormap for different colors
#     color_values = plt.cm.get_cmap(colors)(
#         np.linspace(0, 1, len(most_imp_features)))

#     # Create a horizontal bar chart
#     plt.figure(figsize=(10, 6))
#     plt.barh(range(len(most_imp_features)), most_imp_features.Importance,
#              color=color_values, edgecolor='black', linewidth=1.2)

#     # Customize the plot
#     plt.yticks(range(len(most_imp_features)),
#                most_imp_features.Feature, fontsize=14)
#     plt.xlabel('Importance')
#     plt.title('Most important features - ' + best_one)

#     # Show the plot
#     plt.show()


def data_calssification_part():
    file_path = 'diabetes_risk_prediction_dataset.csv'
    df = load_data(file_path)
    preprocess_data_for_classifier(df)

    x, y = df.shape
    print('\n\n\nnumber of rows : '+str(x)+'\nnumber of columns : '+str(y))

    features, goal = determine_features_and_goal(df)

    target = goal.unique().tolist()

    print("\n\n\n\n\n----------------------------------------------------------------------------------------------------------------------")
    print("\n\n The target is : "+str(target))
    print("\n\n\n The number of target is : "+str(len(target)))

    featuresList = list(features)
    print("\n\n\n Our Features is:\n\n")
    for i in range(0, len(featuresList), 4):
        print(featuresList[i:i+4])

    # splitting our data to testing and training data
    X_train, X_test, y_train, y_test = split_data(
        features, goal, test_size=0.3, random_state=3)

    X_train = label_encode_categorical_features(X_train)
    X_test = label_encode_categorical_features(X_test)

    print("\n\n\n number of training data  : " + str(len(X_train)))
    print("\n\n\n number of testing data  : " + str(len(X_test)))

    # KNN
    K = round(np.sqrt(x))
    y_pred_knn = apply_knn_classifier(K, X_train, X_test, y_train)

    A_res, P_res, R_res, con = calculate_performance(y_test, y_pred_knn)
    print("\n\n\nKNN report\n", classification_report(y_test, y_pred_knn,
          labels=target, target_names=target, zero_division=1))
    print("KNN Accuracy:", A_res, '%')
    print("KNN Precision:", P_res, '%')
    print("KNN Recall:", R_res, '%')
    print("KNN Confusion Matrix:\n", con)

    # Naive Bayes
    y_pred_nb = apply_naive_bayes_classifier(X_train, X_test, y_train)
    A_res, P_res, R_res, con = calculate_performance(y_test, y_pred_nb)
    print("\n\n\nNaive report\n\n\n", classification_report(y_test, y_pred_nb,
          labels=target, target_names=target, zero_division=1))
    print("NaiveBayesAccuracy: ", A_res, '%')
    print("NaiveBayes Precision:", P_res, '%')
    print("NaiveBayes Recall:", R_res, '%')
    print("Naive Confusion Matrix:\n", con)

    # Decision Tree
    y_pred_dt, dt = apply_decision_tree_classifier(X_train, X_test, y_train)
    A_res, P_res, R_res, con = calculate_performance(y_test, y_pred_dt)
    print("\n\n\nDecision Tree report\n\n\n", classification_report(y_test, y_pred_dt,
          labels=target, target_names=target, zero_division=1))
    print("Decision Tree Accuracy: ", A_res, '%')
    print("Decision Tree Precision:", P_res, '%')
    print("Decision Tree Recall:", R_res, '%')
    print("DT Confusion Matrix:\n", con)

    # Use this code to draw the decision tree and save it in tree.png
    F = list(features)
    dot_data = tree.export_graphviz(dt, feature_names=F, class_names=target,
                                    filled=True, rounded=True, special_characters=True, out_file=None,)
    graph = graphviz.Source(dot_data)
    graph.format = "png"
    graph.render("Tree")

    # Define the best classifier based on accuracy
    best_classifier = best_classifier_function(
        X_train, X_test, y_train, y_test)
        # Assuming X_train and dt are available in the global scope
    feature_names = X_train.columns
    feature_imports = dt.feature_importances_

    # Create a DataFrame with feature names and importances, then select the top 10
    most_imp_features = pd.DataFrame([f for f in zip(feature_names, feature_imports)], columns=[
                                     "Feature", "Importance"]).nlargest(10, "Importance")
    most_imp_features.sort_values(by="Importance", inplace=True)

    # Use a colormap for different colors
    color_values = plt.cm.get_cmap('viridis')(
        np.linspace(0, 1, len(most_imp_features)))

    # Create a horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(most_imp_features)), most_imp_features.Importance,
             color=color_values, edgecolor='black', linewidth=1.2)

    # Customize the plot
    plt.yticks(range(len(most_imp_features)),
               most_imp_features.Feature, fontsize=14)
    plt.xlabel('Importance')
    plt.title('Most important features - ' + best_classifier)

    # Show the plot
    plt.show()
