import numpy as np
import pandas as pd


def save_fuzzy_system_results(results, withdrawal_percentage,
                              hour, transactions_per_day,
                              transactions_per_month):
    for method in results[0].keys():
        output_values = [entry[method] for entry in results]
        output_filename = f"Project_2/outputs/output_{method}.csv"

        # Save input and output variables to a CSV file
        data = np.column_stack((withdrawal_percentage, hour,
                                transactions_per_day,
                                transactions_per_month, output_values))
        header = "withdrawal_percentage,hour,transactions_per_day," + \
            "transactions_per_month,output"
        np.savetxt(output_filename, data, delimiter=',', header=header, comments='')

        print(f"Saved results for {method} in {output_filename}")

    return results


def read_synthetic_data(filename, current_path):
    folders = current_path.split("/")
    if folders[-1] == "Code" and folders[-2] == "Project_2":
        path = "/".join(folders[:-1]) + "/outputs" + "/" + filename
    elif folders[-1] == "Project_2":
        path = "/".join(folders[:]) + "/outputs" + "/" + filename
    elif folders[-1] == "Artificial-Intelligence":
        path = "/".join(folders[:]) + "/Project_2" + "/outputs" + "/" + filename
    else:
        raise ValueError("You are not in the right folder to read the default data")

    try:
        data = pd.read_csv(path)
        return data
    except FileNotFoundError:
        print("Could not find file using default data.")
        exit(1)
