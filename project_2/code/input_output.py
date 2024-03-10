import numpy as np
import os
import pandas as pd


def __path_to_data() -> str:
    """
    A function to determine the path to the data based on the current folder structure.

    :return: The path to the data.
    """
    current_path = os.getcwd()
    folders = current_path.split("/")
    # The following if statement allows the user to run the script in any folder in the project 2.
    if folders[-1] == "code" and folders[-2] == "project_2":
        path = "/".join(folders[:-1]) + "/outputs" + "/"
    elif folders[-1] == "project_2":
        path = "/".join(folders[:]) + "/outputs" + "/"
    elif folders[-1] == "Artificial-Intelligence":
        path = "/".join(folders[:]) + "/project_2" + "/outputs" + "/"
    else:
        raise ValueError("You are not in the right folder to read the default data")

    return path


def save_fuzzy_system_results(results: list[dict], withdrawal_percentage: np.ndarray, hour: np.ndarray,
                              transactions_per_day: np.ndarray, transactions_per_month: np.ndarray,
                              location: str = None):
    """
    Save fuzzy system results to a CSV file, with the option to specify a location for the output files. If no location
    is specified, the output files will be saved in the 'outputs' folder inside the project 2.
    Parameters:
        results: a list of dictionaries containing the fuzzy system results. Each dictionary contains the results for
            each defuzzification method that was used to run the fuzzy system.
        withdrawal_percentage: an array of withdrawal percentages
        hour: an array of hours
        transactions_per_day: an array of transactions per day
        transactions_per_month: an array of transactions per month
        location: an optional string specifying the location for the output files
    """
    for method in results[0].keys():
        output_values = [entry[method] for entry in results]
        if location:
            output_filename = f"{location}/output_{method}.csv"
        else:
            path = __path_to_data()
            output_filename = f"{path}/output_{method}.csv"

        # Save input and output variables to a CSV file
        data = np.column_stack((withdrawal_percentage, hour,
                                transactions_per_day,
                                transactions_per_month, output_values))
        header = "withdrawal_percentage,hour,transactions_per_day," + \
            "transactions_per_month,output"
        np.savetxt(output_filename, data, delimiter=',', header=header, comments='')

        print(f"Saved results for {method} in {output_filename}")


def read_data(filename: str = None, custom_path_to_data: str = None):
    """
    A function to read data from either a default file or a custom file path.
    It takes in two optional parameters: filename (default data) and custom_path_to_data (custom data).

    Parameters:
        filename: str, the name of the default data file
        custom_path_to_data: str, the path to the custom data file

    Returns:
        data: pd.DataFrame, the data read from the specified file
    """
    # Check if it is default data
    if filename:
        path = __path_to_data() + filename
    # Check if it is custom data
    elif custom_path_to_data:
        path = custom_path_to_data
    else:
        raise ValueError("You must specify either filename or custom_path_to_data")

    data = pd.read_csv(path)

    return data
