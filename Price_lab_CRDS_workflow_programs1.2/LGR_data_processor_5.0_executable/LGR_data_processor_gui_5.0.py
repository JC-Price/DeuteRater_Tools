# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 09:59:39 2024

@author: cniels21
"""


import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import matplotlib
import re
matplotlib.use('Agg')

def collapse_dataframe(df):
    # Group by the first column (Sample Name) and calculate the mean of each group
    collapsed_df = df.groupby('Sample Name', as_index=False)['Asymptote'].mean()
    
    return collapsed_df

def extract_before_tech(input_string):
    if "_tech" in input_string:
        return input_string.split("_tech")[0]
    else:
        return input_string
    
def make_valid_file_name(input_str, replacement_char='_'):
    """
    Convert a string into a valid file name by replacing invalid characters.
    
    Args:
        input_str (str): The input string to convert.
        replacement_char (str, optional): The character to use as a replacement for invalid characters.
    
    Returns:
        str: The valid file name.
    """
    # Define a regular expression pattern to match invalid characters in file names
    invalid_chars_pattern = r'[\/:*?"<>|]'

    # Replace invalid characters with the replacement character
    valid_file_name = re.sub(invalid_chars_pattern, replacement_char, input_str)

    return valid_file_name
def choose_input_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    input_file_entry.delete(0, tk.END)  # Clear any previous input
    input_file_entry.insert(0, file_path)  # Set the selected file path

def choose_output_location():
    folder_path = filedialog.askdirectory()
    output_location_entry.delete(0, tk.END)  # Clear any previous input
    output_location_entry.insert(0, folder_path)  # Set the selected folder path

def process_data():
    #function definitions
    def load_csv(file_path):
        if file_path:
            df = pd.read_csv(file_path)
            df = df.dropna()
            return df
        

    def separate_df(df, set_size):
        num_rows = len(df)
        overlapping_dfs = []

        for i in range(0, num_rows, set_size):
            sub_df = df.iloc[i:i+set_size].copy()
            overlapping_dfs.append(sub_df)
        
        for i in range(1, len(overlapping_dfs)):
            overlapping_dfs[i] = pd.concat([overlapping_dfs[i-1].tail(1), overlapping_dfs[i]])
        
        return overlapping_dfs

    def generate_minutes_left_out(df):
        # Convert the 'Time code' column to datetime objects <-- finish this and find out if there is a coorelation between minutes sat out and divation
        df['Time Code'] = pd.to_datetime(df['Time Code'])
        # Find the minimum time code in the DataFrame
        min_time_code = df['Time Code'].min()
        # Calculate the time difference in minutes and store it in a 'Minutes_sat_out' column
        df['Minutes_sat_out'] = (df['Time Code'] - min_time_code).dt.total_seconds() / 60
        return df



    def add_numbered_column(dataframe):
        dataframe = dataframe.reset_index(drop=True)
        dataframe['numbered'] = dataframe.index + 1
        return dataframe

    def exponential_growth_with_asymptote(x, a, b, asymptote):
        x_array = np.array(x)  # Convert x_values to a numpy array
        return asymptote - (asymptote - a) * np.exp(-b * x_array)


    def fit_exponential_growth_with_asymptote_and_plot(x_values, y_values, sample_name, output_folder):
        sample_name = make_valid_file_name(sample_name)
        p0 = [y_values[0], 0.1, y_values[-1]]  # Initial parameter guesses: initial value, growth rate, asymptote
        worked = False
        try:
            params, _ = curve_fit(exponential_growth_with_asymptote, x_values, y_values, p0=p0)
            asymptote = params[2]
            x_fit = np.linspace(min(x_values), max(x_values), 100)
            y_fit = exponential_growth_with_asymptote(x_fit, *params)
            worked = True
        except:
            print('could not fit exponential curve')
            worked = False
            asymptote = 0
        
        # Calculate Y-value range
        y_range = f"{abs(max(y_values) - min(y_values)):5e}"
    
        # Calculate R-squared value
        y_mean = np.mean(y_values)
        
        if worked:
            y_predicted = exponential_growth_with_asymptote(x_values, *params)
            ss_total = np.sum((y_values - y_mean) ** 2)
            ss_residual = np.sum((y_values - y_predicted) ** 2)
            r_squared = round(1 - (ss_residual / ss_total), 5)
        else:
            r_squared = 0
    
        # Create the output folder if it doesn't exist
        output_folder_path = os.path.join(output_folder, "Asymptote calculation graphs")
        os.makedirs(output_folder_path, exist_ok=True)
    
        # Generate and save the plot
        plt.close()
        plt.figure(figsize=(8, 6))
        if worked:
            plt.plot(x_fit, y_fit, label="Fitted Exponential Curve")
        plt.scatter(x_values, y_values, color='red', label=f"Data Points, R^2:{r_squared}, Y range:{y_range}")
        if r_squared >= .95:
            plt.axhline(y=asymptote, color='gray', linestyle='dashed', label="Asymptote")
        
        plt.xlabel("Run #")
        plt.ylabel("D/H")
        plt.title("Fitted Exponential Curve: {}".format(sample_name))
        plt.legend()
        plt.grid(True)
    
        # Save the plot with the sample name in the output folder
        plot_filename = os.path.join(output_folder_path, f"{sample_name}_plot.png")
        plt.savefig(plot_filename)
        print(sample_name)
        
        # Close the plot to release resources
        plt.close()
    
        return asymptote, r_squared, y_range

    def save_to_csv(df, name, output_folder):
         file_path = os.path.join(output_folder, name)
         if file_path:
             df.to_csv(file_path, index=False)
             
                     
    def generate_tech_reps_df(df, list_of_dfs, output_folder):
        # Update the DataFrames in the list
        for i, df in enumerate(list_of_dfs):
            list_of_dfs[i] = add_numbered_column(df)
            
        output_data= []
        
        
        for i in list_of_dfs:
            try:
                asymptote, r_squared, y_range = fit_exponential_growth_with_asymptote_and_plot(i["numbered"].tolist(), i["D/H"].tolist(), i.iloc[1, 5], output_folder)
                if r_squared >= .95:
                    status = "Good"
                    output_data.append({
                        "Sample Name": i.iloc[1, 5],
                        "Asymptote": asymptote,
                        "R-squared": r_squared, "Y range":y_range, 'Minutes_left_out': i.Minutes_sat_out[1],
                        "Status" : status})
                else:
                    last_three_values = i['D/H'][-3:]
                    average_last_three = last_three_values.mean()
                    status = "Lower exponential conformity: Assymptote instead the average of last three measurements"
                    output_data.append({
                        "Sample Name": i.iloc[1, 5],
                        "Asymptote": average_last_three,
                        "R-squared": r_squared, "Y range":y_range, 'Minutes_left_out': i.Minutes_sat_out[1],
                        "Status" : status})
            except:
                print('failed')
                last_three_values = i['D/H'][-3:]
                average_last_three = last_three_values.mean()
                output_data.append({
                    "Sample Name": i.iloc[1, 5],
                    "Asymptote": average_last_three,
                    "R-squared": "Failed to fit to exponential equation", "Y range":"Failed", 'Minutes_left_out': i.Minutes_sat_out[1],
                    "Status" : "Assymptote instead the average of last three measurements"
                })
        
        tech_reps_df = pd.DataFrame(output_data).sort_values(by = "Sample Name")
        return tech_reps_df

    # Fit a linear equation to the averaged points
    def linear_equation(x, m, b):
        return m * x + b

    def standards_df_generation(tech_reps_df, file_name, output_folder, standards):
        standards_df = tech_reps_df[tech_reps_df['Sample Name'].str.startswith('Curve')].reset_index(drop=True)
        print(standards_df)
        #in case full standard curves were mistakenly run multiple times (this happened with our 3/4 biorep_1)
        collapsed_standards_df = collapse_dataframe(standards_df)
        # Define the % deuterium levels
        deuterium_levels = standards
        
        # Initialize empty lists to store calculated values
        curve_averages = []
        curve_std_deviations = []
        
        # Iterate through each set of items across curves
        for i in range(len(deuterium_levels)):
            curve_values = collapsed_standards_df.loc[i::len(deuterium_levels), "Asymptote"]
            avg = np.mean(curve_values)
            std_dev = np.std(curve_values)
            
            curve_averages.append(avg)
            curve_std_deviations.append(std_dev)
        
        
        
        params, _ = curve_fit(linear_equation, deuterium_levels, curve_averages)
        
        # Calculate R-squared value
        residuals = np.array(curve_averages) - linear_equation(np.array(deuterium_levels), *params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((np.array(curve_averages) - np.mean(curve_averages))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        
        
        
        # Create Standard Curve Graph
        plt.figure(figsize=(10, 6))
        plt.errorbar(deuterium_levels, curve_averages, yerr=curve_std_deviations, marker='o', linestyle='-', capsize=5)
        plt.plot(deuterium_levels, linear_equation(np.array(deuterium_levels), *params))
        plt.xlabel('D/H')
        plt.ylabel('Asymptote')
        plt.title('Standard Curve')
        plt.legend()
        
        
        # Display linear equation and R-squared value on the graph
        equation_text = f'y = {params[0]:.6f}x + {params[1]:.6f}'
        r_squared_text = f'$R^2$ = {r_squared:.6f}'
        plt.annotate(equation_text, xy=(0.1, 0.0004), fontsize=10, color='red')
        plt.annotate(r_squared_text, xy=(0.1, 0.00038), fontsize=10, color='red')
        
        plt.grid(True)
        plt.show()
        
        plot_filename = os.path.join(output_folder, f"{file_name}_plot.png")
        plt.savefig(plot_filename)
        
        # Create final_standards_output DataFrame
        final_standards_output = pd.DataFrame({
            'D/H': deuterium_levels,
            'Average Asymptote': curve_averages,
            'Standard Deviation': curve_std_deviations,
            'R-squared': [r_squared] * len(deuterium_levels)
        })
        return final_standards_output, params[0], params[1], standards_df

    #there is some janky math here... done too late at night lol
    def samples_df_generation(samples_df, m, b):
        samples_df['Sample'] = samples_df['Sample Name'].apply(lambda x: extract_before_tech(x))
        
        samples = list(set(samples_df['Sample Name'].str.split('_tech').str[0]))
        averages = []
        std_devs = []
        
        D_concentrations = []
        D_concentrations_std_dev = []
        av_time_left_outs = []
        std_dev_time_left_outs = []
        #sample_df_list = []
        
        for sample in samples:
            # Filter the DataFrame to include only rows with the current sample
            sample_df = samples_df[samples_df['Sample'] == sample]
            
            # Calculate the average for 'Asymptote' and 'R-squared' columns
            avg_asymptote = sample_df['Asymptote'].mean()
            std_dev = sample_df['Asymptote'].std()
            Local_D_concentrations = sample_df['Asymptote'].apply(lambda x: (x-b)/m)
            #print(len(Local_D_concentrations))
            D_concentration = Local_D_concentrations.mean()
            D_concentration_std_dev = Local_D_concentrations.std()
            av_time_left_out = sample_df["Minutes_left_out"].mean()
            std_dev_time_left_out = sample_df["Minutes_left_out"].std()
            
            
            
            # Append the average values to the 'averages' list
            averages.append((avg_asymptote))
            std_devs.append((std_dev))        
            rel_std_dev_DH = [(std_dev / avg) * 100 for avg, std_dev in zip(averages, std_devs)]
            
            D_concentrations.append((D_concentration))
            D_concentrations_std_dev.append(D_concentration_std_dev)
            rel_std_dev_D_conc = [(std_dev / conc) * 100 for conc, std_dev in zip(D_concentrations, D_concentrations_std_dev)]
            av_time_left_outs.append(av_time_left_out)
            std_dev_time_left_outs.append(std_dev_time_left_out)

        
        # Assuming you have three lists: samples, averages, and std_devs
        samples_data = {'Samples': samples, 'Average D/H': averages, 'Std Dev D/H': std_devs, 'Rel Std Dev D/H':rel_std_dev_DH, 'Average D2O%': D_concentrations, 'Std Dev D2O%':D_concentrations_std_dev, 'Rel Std Dev D2O%':rel_std_dev_D_conc, 'Average minutes left out':av_time_left_outs, 'Std Dev minutes left out':std_dev_time_left_outs}
        # Create a DataFrame from the data dictionary
        final_samples_output = pd.DataFrame(samples_data)
        try:
            final_samples_output['Days'] = final_samples_output['Samples'].str.extract(r'D(\d+\.\d+|\d+)').astype(float)
        except:
            print(final_samples_output['Samples'].tolist())
            
        print(f'Columns for final_samples_output{final_samples_output.columns}')
        
        final_samples_output.insert(1, 'Days', final_samples_output.pop('Days'))
        return final_samples_output

    def plot_time_vs_enrichment(dataframe, genotype, name, output_folder):
        # Extract data from the DataFrame
        dataframe = dataframe.reset_index(drop=True)
        days = dataframe['Days']
        average_d2o_percent = dataframe['Average D2O%']
        std_dev_d2o_percent = dataframe['Std Dev D2O%']
        samples = dataframe['Samples']

        # Extract the title from the first row of the "Samples" column
        title = samples.iloc[0][:6]

        # Create the plot
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        plt.errorbar(days, average_d2o_percent, yerr=std_dev_d2o_percent, fmt='o', capsize=5)

        # Set axis labels and title
        plt.xlabel('Days')
        plt.ylabel('Average D2O%')
        plt.title(f'{title} {genotype}')  # Use the extracted title

        # Label each data point with the corresponding "Sample" value
        for i, sample in enumerate(samples):
            plt.annotate(sample, (days[i], average_d2o_percent[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize = 5)

        # Show the plot
        plt.grid(True)
        plt.show()
        plot_filename = os.path.join(output_folder, f"{name}_{genotype}_plot.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')


    def split_dataframe_by_prefix_and_baseline(dataframe, identifiers, sub_identifiers):
        # Check if 'treatment' is present in the 'Samples' column
        #identifiers = ['BlaJ']
        #sub_identifiers = ['treatment', 'saline']
        
        
        # Initialize variables
        identifiers_df_list = []
        subidentifiers_df_list_of_lists = []
        identifiers_counter = -1
        for i in identifiers:
            subidentifiers_df_list = []
            identifiers_counter+=1
            sub_identifier_counter = -1     
            if any(dataframe['Samples'].str.contains(i)):
                identifiers_df_list.append(dataframe[dataframe['Samples'].str.contains(i)])
                for j in sub_identifiers:
                    sub_identifier_counter+=1
                    if any(dataframe["Samples"].str.contains(j)):
                        subidentifiers_df_list.append(identifiers_df_list[identifiers_counter][identifiers_df_list[identifiers_counter]['Samples'].str.contains(j)])
            subidentifiers_df_list_of_lists.append(subidentifiers_df_list)
        return identifiers, sub_identifiers, identifiers_df_list, subidentifiers_df_list_of_lists
    
                


    #gui data
    input_file = input_file_entry.get()
    file_name = os.path.basename(input_file)
    output_location = output_location_entry.get()
    output_folder = os.path.join(output_location, file_name[:-4] +'_'+ analysis_label_entry.get()+ '_' + "ouput")
    
    standards = standards_entry.get().split(',')
    standards = [float(value) for value in standards]
    print(standards)
    identifiers = identifiers_entry.get().split(',')
    subidentifiers = subidentifiers_entry.get().split(',')
    
    print("Standard D2O concentrations:", standards)
    print("Identifiers:", identifiers)
    print("Subidentifiers:", subidentifiers)
    
    #script
    df = load_csv(input_file)
    
    #add minutes left out column
    df = generate_minutes_left_out(df)
    
    #splitting dfs up
    global list_of_dfs
    list_of_dfs = separate_df(df, 8)

    #creating technical reps df
    tech_reps_df = generate_tech_reps_df(df, list_of_dfs, output_folder)
    
    save_to_csv(tech_reps_df, "tech_reps_df.csv", output_folder)

    #creating standards df and standard curve formula
    final_standards_output, m, b, standards_df= standards_df_generation(tech_reps_df, 'Standard_Curve',  output_folder, standards)
    
    save_to_csv(final_standards_output, 'final_standards_output.csv', output_folder)
    global samples_df
    samples_df = tech_reps_df[~tech_reps_df['Sample Name'].str.startswith('Curve')]
    #creating samples df
    final_samples_output  = samples_df_generation(samples_df, m, b)
    
    save_to_csv(final_samples_output, 'final_samples_output.csv', output_folder)

    #splitting up final_samples_output into males and females
    identifiers, sub_identifiers, identifiers_df_list, subidentifiers_df_list_of_lists = split_dataframe_by_prefix_and_baseline(final_samples_output, identifiers, subidentifiers)

     # Creating enrichment vs. days curve
    identifier_counter = -1
    for i in subidentifiers_df_list_of_lists:
        identifier_counter +=1
        sub_identifier_counter = -1
        print(len(i))
        for j in i:
            sub_identifier_counter +=1
            plot_time_vs_enrichment(j, sub_identifiers[sub_identifier_counter], identifiers[identifier_counter], output_folder)
       
    
    
    #done
    print('done')

# Create the main window

# Create the main window
root = tk.Tk()
root.title("Price lab LGR Cavity Ring Down Spectroscopy Data Analysis Software")

# Create and pack an entry field for Analysis name
analysis_label = tk.Label(root, text="Title of analysis:")
analysis_label.pack()

analysis_label_entry = tk.Entry(root, width=50)
analysis_label_entry.pack()

# Create and pack a label for input file
input_file_label = tk.Label(root, text="Choose Input File:")
input_file_label.pack()

# Create and pack an entry field for input file
input_file_entry = tk.Entry(root, width=50)
input_file_entry.pack()

# Create and pack a button for choosing input file
choose_input_button = tk.Button(root, text="Choose Input File", command=choose_input_file)
choose_input_button.pack()

# Create and pack a label for output location
output_location_label = tk.Label(root, text="Choose Output Location:")
output_location_label.pack()

# Create and pack an entry field for output location
output_location_entry = tk.Entry(root, width=50)
output_location_entry.pack()

# Create and pack a button for choosing output location
choose_output_button = tk.Button(root, text="Choose Output Location", command=choose_output_location)
choose_output_button.pack()

# Create and pack an entry field for standards
standards_label = tk.Label(root, text="Standard D2O concentrations (comma-delimit):")
standards_label.pack()

# Set default value for the entry field
default_value = "0,0.5,1,2,4,6,10"
standards_entry = tk.Entry(root, width=50)
standards_entry.insert(0, default_value)  # Insert default value at index 0
standards_entry.pack()

# Create and pack an entry field for Identifiers
identifiers_label = tk.Label(root, text="Identifiers (comma-delimit):")
identifiers_label.pack()

identifiers_entry = tk.Entry(root, width=50)
identifiers_entry.pack()

# Create and pack an entry field for Subidentifiers
subidentifiers_label = tk.Label(root, text="Subidentifiers (comma-delimit):")
subidentifiers_label.pack()

subidentifiers_entry = tk.Entry(root, width=50)
subidentifiers_entry.pack()

# Create and pack a button for processing data
process_button = tk.Button(root, text="Process Data", command=process_data)
process_button.pack()

# Start the main loop
root.mainloop()

