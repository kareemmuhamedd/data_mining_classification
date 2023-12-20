import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from data_classification_part.classifier_functions import *
from utils import *
from sklearn import tree
import graphviz
import threading
from data_visualization_part.visualization_data import DataVisualizer



def print_available_columns(data):
    print("\nAvailable Columns:")
    for i, column in enumerate(data.columns):
        print(f"{i}. {column}")


def visualize_individual_column(data, column_choice, data_visualizer):
    if 0 <= column_choice < len(data.columns):
        column = data.columns[column_choice]
        data_visualizer.visualize_individual_column(data, column)


def visualize_histogram(preprocessed_data, data_visualizer):
    try:
        data_to_visualize = (
            preprocessed_data
            if int(input("\nChoose type:\n1. header, 2. column values: ")) == 2
            else data
        )
        print_available_columns(data_to_visualize)
        column_choice = int(
            input(
                "Enter the number corresponding to the column you want to visualize: "
            )
        )
        visualize_individual_column(
            data_to_visualize, column_choice, data_visualizer)
    except ValueError:
        print("Invalid input. Please enter either 1 or 2.")
    if not try_another_visualization():
        exit()


def visualize_box_plot(preprocessed_data, data_visualizer):
    try:
        data_to_visualize = (
            preprocessed_data
            if int(input("\nChoose type:\n1. header, 2. column values: ")) == 2
            else data
        )
        print_available_columns(data_to_visualize)
        x_column_choice = int(
            input("Enter the number corresponding to the x-axis column: ")
        )
        y_column_choice = int(
            input("Enter the number corresponding to the y-axis column: ")
        )
        visualize_boxplot(
            data_to_visualize, x_column_choice, y_column_choice, data_visualizer
        )
    except ValueError:
        print("Invalid input. Please enter either 1 or 2.")
    if not try_another_visualization():
        exit()


def visualize_boxplot(data, x_column_choice, y_column_choice, data_visualizer):
    if 0 <= x_column_choice < len(data.columns) and 0 <= y_column_choice < len(
        data.columns
    ):
        x_column = data.columns[x_column_choice]
        y_column = data.columns[y_column_choice]
        data_visualizer.visualize_boxplot(data, x_column, y_column)


def visualize_scatter_plot(preprocessed_data, data_visualizer):
    try:
        data_to_visualize = (
            preprocessed_data
            if int(input("\nChoose type:\n1. header, 2. column values: ")) == 2
            else data
        )
        print_available_columns(data_to_visualize)
        x_column_choice = int(
            input("Enter the number corresponding to the x-axis column: ")
        )
        y_column_choice = int(
            input("Enter the number corresponding to the y-axis column: ")
        )
        visualize_scatterplot(
            data_to_visualize, x_column_choice, y_column_choice, data_visualizer
        )
    except ValueError:
        print("Invalid input. Please enter either 1 or 2.")
    if not try_another_visualization():
        exit()


def visualize_pie_chart_plot(preprocessed_data, data_visualizer):
    try:
        data_to_visualize = (
            preprocessed_data
            if int(input("\nChoose type:\n1. header, 2. column values: ")) == 2
            else data
        )
        print_available_columns(data_to_visualize)
        column_choice = int(
            input(
                "Enter the number corresponding to the column you want to visualize: "
            )
        )
        visualize_pie_chart(data_to_visualize, column_choice, data_visualizer)
    except ValueError:
        print("Invalid input. Please enter either 1 or 2.")
    if not try_another_visualization():
        exit()

def visualize_pie_chart(data, column_choice, data_visualizer):
    if 0 <= column_choice < len(data.columns):
        column = data.columns[column_choice]
        data_visualizer.visualize_pie_chart(data, column)

def try_another_visualization():
    return (
        input("Do you want to try another visualization? (yes/no): ").lower() == "yes"
    )

def visualize_scatterplot(data, x_column_choice, y_column_choice, data_visualizer):
    if 0 <= x_column_choice < len(data.columns) and 0 <= y_column_choice < len(
        data.columns
    ):
        x_column = data.columns[x_column_choice]
        y_column = data.columns[y_column_choice]
        data_visualizer.visualize_scatterplot(data, x_column, y_column)

def load_and_preprocess_data():
    while True:
        try:
            file_path = 'diabetes_risk_prediction_dataset.csv'

            # Define a variable to store the loaded data
            loaded_data = None

            # Function to load data in a separate thread
            def load_data_thread():
                nonlocal loaded_data
                loaded_data= load_data(file_path)

            # Start loading data in a separate thread
            load_data_thread = threading.Thread(target=load_data_thread)
            load_data_thread.start()

            # Wait for the data loading thread to finish
            load_data_thread.join()

            # Print the loaded data
            print("\nDataFrame:")
            print(loaded_data)
            global data
            data = loaded_data

            # Use the same thread for preprocessing the loaded data
            res = preprocess_data_for_visualization(loaded_data)
            return res

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            retry = input("Do you want to try again? (yes/no): ").lower()
            if retry != "yes":
                exit()






def data_visualize_part(preprocessed_data, data_visualizer):
    while True:
        print("\nVisualization Options:")
        print("1. Visualize Histogram ")
        print("2. Visualize Box Plot ")
        print("3. Visualize Scatter Plot ")
        print("4. Visualize Pie Chart ")
        print("5. Back to Main Menu ")

        choice = int(
            input(
                "Enter the number corresponding to the visualization option you want to choose: "
            )
        )

        if choice == 5:
            break
        elif choice == 1:
            visualize_histogram(preprocessed_data, data_visualizer)
        elif choice == 2:
            visualize_box_plot(preprocessed_data, data_visualizer)
        elif choice == 3:
            visualize_scatter_plot(preprocessed_data, data_visualizer)
        elif choice == 4:
            visualize_pie_chart_plot(preprocessed_data, data_visualizer)
        else:
            print("Invalid choice. Please try again.")
