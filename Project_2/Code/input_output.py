import numpy as np

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
