# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 11:45:41 2023

@author: colem
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from datetime import datetime
import pandas as pd
from tkinter import messagebox
import numpy as np
import random



def randomize_df(input_df):
    randomized_df = input_df.sample(frac=1, random_state=random.seed())
    return randomized_df
# Default settings from the provided text file
default_settings = {
    "Sample Volume (nL)": 920,
    "Fill Speed (nL/s)": 1000,
    "Eject Speed (nL/s)": 1000,
    "Flush Strokes": 6,
    "Pullup Delay (ms)": 300,
    "Injection Speed (nL/s)": 1000,
    "Pre Injection Delay (ms)": 50,
    "Post Injection Delay (ms)": 25000,
    "Post Withdrawl Delay (ms)": 10000,
    "Initial Flush Time (ms)": 5000,
    "Purge Fill Time (ms)": 3000,
    "Final Pumpdown Time (ms)": 45000,
    "Septum Vent Time (ms)": 180000,
    "Septum Evacuate Time (ms)": 120000,
    "Shutdown Vent Time (ms)": 180000,
    "Measurements per Sample": 20,
    "Injections per Sample": 8,
    "Start with Samples or Injections (Samp = 0, Inj = 1)": 1,
    "Run Standards after X": 4,
    "Standard Interleave Type (Group = 0, One per Sample = 1)": 1
}

section_headers = {
    "Fill Options": [
        "Sample Volume (nL)",
        "Fill Speed (nL/s)",
        "Eject Speed (nL/s)",
        "Flush Strokes",
        "Pullup Delay (ms)"
    ],
    "Injection Options": [
        "Injection Speed (nL/s)",
        "Pre Injection Delay (ms)",
        "Post Injection Delay (ms)",
        "Post Withdrawl Delay (ms)"
    ],
    "Advanced Setup Options": [
        "Initial Flush Time (ms)",
        "Purge Fill Time (ms)",
        "Final Pumpdown Time (ms)"
    ],
    "Misc. Settings": [
        "Septum Vent Time (ms)",
        "Septum Evacuate Time (ms)",
        "Shutdown Vent Time (ms)",
        "Measurements per Sample"
    ],
    "Run List Settings": [
        "Injections per Sample",
        "Start with Samples or Injections (Samp = 0, Inj = 1)",
        "Run Standards after X",
        "Standard Interleave Type (Group = 0, One per Sample = 1)"
    ]
}



class SettingsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LGR Cavity Ring-Down Spectrometer settings")

        self.settings = {key: tk.StringVar(value=str(value)) for key, value in default_settings.items()}
        self.include_standards = tk.BooleanVar(value=False)
        self.samples_imported = False  # Initialize samples_imported attribute
        self.standards_imported = False  # Initialize standards_imported attribute
        self.samples_df = None
        self.standards_df = None

        self.create_gui()

    def create_gui(self):
        for section, keys in section_headers.items():
            section_frame = ttk.LabelFrame(self.root, text=section)
            section_frame.pack(padx=10, pady=5, side="left", fill="both")

            for key in keys:
                label = ttk.Label(section_frame, text=key)
                label.pack(anchor='w', padx=10, pady=2)

                entry = ttk.Entry(section_frame, textvariable=self.settings[key])
                entry.pack(fill='x', padx=10, pady=2)

        standards_checkbox = ttk.Checkbutton(self.root, text="Included standards", variable=self.include_standards, command=self.update_import_standards_button)
        standards_checkbox.pack(padx=10, pady=5)

        save_button = ttk.Button(self.root, text="Save Configuration", command=self.save_configuration)
        save_button.pack(padx=10, pady=10)

        reset_button = ttk.Button(self.root, text="Reset to Default", command=self.reset_to_default)
        reset_button.pack(padx=10, pady=10)

        self.import_samples_button = ttk.Button(self.root, text="Import Samples .csv", command=self.import_samples_csv)
        self.import_samples_button.pack(padx=10, pady=10)

        self.import_standards_button = ttk.Button(self.root, text="Import Standards .csv", command=self.import_standards_csv)
        self.import_standards_button.pack(padx=10, pady=10)
        self.update_import_standards_button()
        
    def get_time(self):
        now = datetime.now()
        return now.strftime("%Y %b %d %H:%M:%S")

    def save_configuration(self):
        
        if not self.samples_imported:
            messagebox.showerror("Error", "Please import samples before saving the configuration.")
            return
    
        if self.include_standards.get() and not self.standards_imported:
            messagebox.showerror("Error", "Please import standards before saving the configuration.")
            return
    
        filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if filename:
            with open(filename, "w", newline= '\n') as file:
                file.write("#LGR ICOS V09.09.11.H2O_4ISO.LWIA Water Isotope Configuration File -- {}\n#Advanced Injection Settings".format(self.get_time()))
                counter = 1
                for section, keys in section_headers.items():
                    file.write(f"\n#{section}\n")
                    
                    for key in keys:
                        if counter == 4:
                            value = self.settings[key].get()
                            file.write(f"{key}:\t\t\t{value}\n")
                        
                        elif counter == 7 or counter == 8 or counter == 9 or counter == 10 or counter == 12 or counter == 14 or counter == 15 or counter == 16 or counter == 20:
                            value = self.settings[key].get()
                            file.write(f"{key}:\t{value}\n")
                        
                        elif counter == 17 or counter == 19:
                            value = self.settings[key].get()
                            file.write(f"{key}:\t\t\t\t\t\t{value}\n")
                            
                        else:
                            value = self.settings[key].get()
                            file.write(f"{key}:\t\t{value}\n")
                        counter += 1
                file.write("\n@Begin Sample Run List\n")
                file.write(randomize_df(self.samples_df).to_csv(index=False, lineterminator='\n'))
                file.write('@End Sample Run List\n\n@Begin Standard Run List\n')
                
                if self.standards_imported:
                    file.write(randomize_df(self.standards_df).to_csv(index=False, lineterminator='\n'))
                file.write('@End Standard Run List\n\n@Occupied Vial Positions\n')
                
                # Create a 4x54 array filled with zeros
                zeros_array = np.zeros((4, 54))
                # Iterate through self.samples_df and self.standards_df
                for _, row in self.samples_df.iterrows():
                    tray = row['Tray']
                    position = row['Position']
                    zeros_array[tray-1, position-1] += 1
                
                if self.standards_imported:
                    for _, row in self.standards_df.iterrows():
                        tray = row['Tray']
                        position = row['Position']
                        zeros_array[tray-1, position-1] += 1
                    
                
                for row in zeros_array:
                    row_str = " ".join(str(int(value)) for value in row)
                    file.write(row_str + "\n")
                file.write("\n")
                    
                file.write("@Begin Available Samples List\n")
                file.write(self.samples_df.to_csv(index=False, lineterminator='\n'))
                file.write('@End Available Samples List\n\n@Begin Available Standards List\n')
                
                if self.standards_imported:
                    file.write(self.standards_df.to_csv(index=False, lineterminator='\n'))
                file.write('@End Available Standards List\n')
                
                
                
                               
                
                    
                
                        
    

    def reset_to_default(self):
        for key, value in default_settings.items():
            self.settings[key].set(str(value))
    def import_samples_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])

        if file_path:
            try:
                df = pd.read_csv(file_path)
                if set(df.columns) == {"Name", "S/N", "Tray", "Position"}:
                    messagebox.showinfo("CSV Loaded", "CSV loaded successfully!")
                    self.samples_df = df  # Store the DataFrame
                else:
                    messagebox.showerror("Incorrect Format", "Incorrect column headers. Please select a .csv file with correct headers.")
            except pd.errors.EmptyDataError:
                messagebox.showerror("Error", "Selected file is empty or not in a valid CSV format.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
        self.samples_imported = True
                
    def import_standards_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])

        if file_path:
            try:
                df = pd.read_csv(file_path)
                if set(df.columns) == {"Name", "S/N", "Tray", "Position"}:
                    messagebox.showinfo("CSV Loaded", "CSV loaded successfully!")
                    self.standards_df = df  # Store the DataFrame
                else:
                    messagebox.showerror("Incorrect Format", "Incorrect column headers. Please select a .csv file with correct headers.")
            except pd.errors.EmptyDataError:
                messagebox.showerror("Error", "Selected file is empty or not in a valid CSV format.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
        self.standards_imported = True
        
    def update_import_standards_button(self):
        if self.include_standards.get():
            self.import_standards_button.config(state=tk.NORMAL)
        else:
            self.import_standards_button.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = SettingsApp(root)
    root.mainloop()
