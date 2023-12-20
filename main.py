from sklearn.metrics import classification_report
from data_classification_part.classifier_functions import *
from utils import *
from data_visualization_part.visualization_data import DataVisualizer
from data_visualization_part.visualization_core import load_and_preprocess_data,data_visualize_part
from data_classification_part.classification_core import data_calssification_part









if __name__ == '__main__':
    preprocessed_data = load_and_preprocess_data()
    data_visualizer = DataVisualizer()

    while True:
        print("\nMain Menu:")
        print("1. Data Classification")
        print("2. Data Visualization")
        print("3. Exit")

        choice = int(input("Enter the number corresponding to your choice: "))

        if choice == 3:
            break
        elif choice == 1:
            data_calssification_part()
        elif choice == 2:
            data_visualize_part(preprocessed_data, data_visualizer)
        else:
            print("Invalid choice. Please try again.")





