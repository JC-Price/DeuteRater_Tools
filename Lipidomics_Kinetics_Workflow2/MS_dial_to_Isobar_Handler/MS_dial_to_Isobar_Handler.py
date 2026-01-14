
"""
---------------------------------------
This program prepares MS-Dial Intensity + RT exports for downstream Isobar Handler / DeuteRater pipelines.

What this file does
-------------------
• Provides a tiny Tkinter GUI that prompts for:
    - an RT CSV and an MS-Dial output CSV (both read with skiprows=4)
    - a destination filename to write a single merged CSV.

• Merges MS-Dial Intensity + Retention time tables on `Alignment ID`
  (inner join), normalizes and synthesizes fields, and converts MS-Dial
  conventions into the DeuteRater syntax. 

"""

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import re
from collections import defaultdict

drop_bad_scores = True
merge_deuterium = True

def merge_deuterium_to_hydrogen(formula: str) -> str:
    """
    Merge any deuterium (D) atoms into hydrogen (H) in a chemical formula.
    Example:
        C16H2D7O2 → C16H9O2
        C38H70D4NO8P → C38H74NO8P
    """
    # Replace [2H] style if it appears
    formula = formula.replace("[2H]", "D")

    # Use your existing parser
    elements = list_chemical_formula(formula)

    # Merge D into H
    if "D" in elements:
        elements["H"] = elements.get("H", 0) + elements.pop("D")

    return format_chemical_formula(elements)




def deduplicate_best_ms_flag(
    df: pd.DataFrame,
    id_col: str = "Lipid Unique Identifier",
    flag_col: str = "MS Dial Flag"
) -> pd.DataFrame:
    """
    Drop duplicate lipids, keeping the row with the *best* MS Dial Flag
    in each group.

    “Best” is defined by the ordering  
        Normal/blank (None or NaN)  <  "low score"  <  "no MS2"

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least `id_col` and `flag_col`.
    id_col : str, default "Lipid Unique Identifier"
        Column that uniquely identifies each lipid.
    flag_col : str, default "MS Dial Flag"
        Column with flag values ("low score", "no MS2", or blank).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed and only the highest-quality
        entry for each lipid retained.
    """
    # 1. Build a numeric rank: lower is better
    priority = {"low score": 1, "no ms2": 2}
    rank = (
        df[flag_col]
        .astype(str)            # handle None/NaN gracefully
        .str.lower()
        .map(priority)
        .fillna(0)              # NaN → 0  (Normal/blank = best)
    )

    # 2. Pick the lowest-ranked row per lipid
    best_idx = rank.groupby(df[id_col]).idxmin()
    
    # 3. Return a copy without the helper Series
    return df.loc[best_idx].reset_index(drop=True)


def strip_ms_flags(text: str) -> str:
    """
    Remove the leading flag prefixes “low score:” and “no MS2:” (case-insensitive)
    from a string and return the cleaned result.
    
    Examples
    --------
    >>> strip_ms_flags("low score: PC 34:2")
    'PC 34:2'
    >>> strip_ms_flags("no MS2: TG 52:3")
    'TG 52:3'
    >>> strip_ms_flags("Low Score: no MS2: DG 36:1")
    'DG 36:1'
    """
    pattern = r"^\s*(low score:|no ms2:)\s*"
    # Keep stripping until none of the prefixes remain at the start
    while re.match(pattern, text, flags=re.I):
        text = re.sub(pattern, "", text, count=1, flags=re.I)
    return text.strip()

def convert_adduct(adduct):
    #positive adducts
    if adduct == '[M+H]+':
        return 'M+H'
    elif adduct == '[M+H-H2O]+':
        return 'M+H-[H2O]'
    elif adduct == '[M+NH4]+':
        return 'M+NH4'
    elif adduct == '[M+Na]+':
        return 'M+Na'

    elif adduct == '[M+Na-H2O]+':
        return 'M+Na-[H2O]'
    elif adduct == '[M+NH4-H2O]+':
        return 'M+NH4-[H2O]'
    
    
    #negative adducts
    elif adduct =='[M-H]-':
        return 'M-H'
    elif adduct == '[M-H2O-H]-':
        return 'M-H-[H2O]' 
    elif adduct == '[M+CH3COO]-':
        return '+C2H3O2-'
    elif adduct =='[M+HCOO]-':
        return '+COOH-'
    elif adduct == '[M+e]-':
        return '+e-'
    else:
        return float('NaN')
    
def list_chemical_formula(formula):
    # Regular expression pattern to match elements and their counts
    formula = formula.replace('[2H]', 'D')
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    if "D" in formula:
        print(formula)
    

    elements_dict = defaultdict(int)
    for (element, count) in matches:
        if count == '':
            count = 1
        else:
            count = int(count)
        elements_dict[element] += count

    return dict(elements_dict)

def format_chemical_formula(elements):
    # Convert elements dictionary back to string format
    formula = ""
    for element, count in elements.items():
        if count == 1:
            formula += element
        elif count == 0:
            pass
        else:
            formula += f"{element}{count}"

    return formula


def adduct_cf(adduct, cf):
    try:
        elements = list_chemical_formula(cf)
        #positive adducts
        if adduct == 'M+H':
            if 'H' in elements:
                elements['H'] += 1
            else:
                elements['H'] = 1
        elif adduct == 'M+H-[H2O]':
            if 'H' in elements:
                elements['H'] -= 1
            else:
                elements['H'] = -1
            if 'O' in elements:
                elements['O'] -= 1
            else:
                elements['O'] = -1   

        elif adduct == 'M+NH4':
            if 'N' in elements:
                elements['N'] += 1
            else:
                elements['N'] = 1
            if 'H' in elements:
                elements['H'] += 4
            else:
                elements['H'] = 4
        elif adduct == 'M+Na':
            if 'Na' in elements:
                elements['Na'] += 1
            else:
                elements['Na'] = 1
                
        elif adduct == 'M+Na-[H2O]':
            if 'Na' in elements:
                elements['Na'] += 1
            else:
                elements['Na'] = 1
            if 'H' in elements:
                elements['H'] -= 1
            else:
                elements['H'] = -1
            if 'O' in elements:
                elements['O'] -= 1
            else:
                elements['O'] = -1   
                
        elif adduct == 'M+NH4-[H2O]':
            if 'N' in elements:
                elements['N'] += 1
            else:
                elements['N'] = 1
            if 'H' in elements:
                elements['H'] += 2
            else:
                elements['H'] = 2
            if 'O' in elements:
                elements['O'] -= 1
            else:
                elements['O'] = -1   
                
        #negative adducts        
        elif adduct == 'M-H':
            if 'H' in elements:
                elements['H'] -= 1
            else:
                elements['H'] = -1
        
        elif adduct == 'M-H-[H2O]':
            if 'H' in elements:
                elements['H'] -=3
            else:
                elements['H'] = -3
            if 'O' in elements:
                elements['O'] -=1
            else: 
                elements['O'] = -1   
                
        elif adduct == '+C2H3O2-':
            if 'H' in elements:
                elements['H'] += 3
            else:
                elements['H'] = 3
            if 'C' in elements:
                elements['C'] += 2
            else: 
                elements['C'] = 2
            if 'O' in elements: 
                elements['O'] += 2
            else:
                elements['O'] = 2
                
            
        elif adduct == '+COOH':
            if 'H' in elements:
                elements['H'] += 1
            else:
                elements['H'] = 1
            if 'C' in elements:
                elements['C'] += 1
            else:
                elements['C'] = 1
            if 'O'in elements:
                elements['O'] +=2
            else:
                elements['O'] = 2
            
        
                
        return format_chemical_formula(elements)

    except:
        return float('NaN')

    

def modify_data_frame(df, remove_PQCs = False):
    drop_bad_scores = True
    new_df = df
    
    new_df = new_df.dropna(subset = "Formula")

    new_df['Lipid Name'] = new_df['Metabolite name'].apply(lambda x: strip_ms_flags(x))
    

    new_df['Lipid Unique Identifier'] = new_df['Lipid Name'].astype(str) + "_" + new_df['Average Rt(min)'].astype(str)

    new_df['Ontology']= new_df['Ontology']

    new_df['Precursor m/z'] = new_df['Average Mz']

    new_df['Precursor Retention Time (min)'] = new_df['Average Rt(min)']
    
    new_df['Precursor Retention Time (sec)'] = new_df['Precursor Retention Time (min)']*60

    new_df['Identification Charge'] = 1

    new_df['LMP'] = ''

    new_df['HMP'] = ''
    
    new_df['Formula'] = new_df['Formula'].replace(r'\[2H\]', 'D', regex=True)
    
    new_df['cf'] = new_df['Formula']

    new_df['neutromers_to_extract'] = 3

    new_df['literature_n'] =''

    new_df['Matched_Results_Analysis'] = ''

    new_df['Matched_Details_Replicates_Used'] =''

    new_df['Adduct'] = new_df['Adduct type'].apply(lambda x:convert_adduct(x))

    if merge_deuterium:
        new_df['Formula'] = new_df['Formula'].apply(merge_deuterium_to_hydrogen)
    new_df['Adduct_cf'] = new_df.apply(lambda row: adduct_cf(row['Adduct'],row['cf']),axis=1)
    new_df = new_df.drop(columns=["Post curation result"])
    new_df['Adducted_Name'] = new_df['Lipid Name'] + '(' + new_df['Adduct'].fillna('') + ')'

    new_df = new_df.dropna()

    drop_conditions = new_df['Metabolite name'].str.contains('no MS2:|low score:', case=False, na=False)


    if drop_bad_scores:
        new_df = new_df[~drop_conditions]
    
        new_df.reset_index(drop = True, inplace=True)
        
        new_df['MS Dial Flag'] = 'Normal'
        
    else:
        
        def find_low_scores(string):
            if 'MS2' in string:    
                return 'no MS2'
            elif 'score' in string:
                return 'low score'
            else: 
                return 'Normal'
        
        new_df['MS Dial Flag'] = new_df['Metabolite name'].apply(lambda x: find_low_scores(x))

    
    # Convert numerical columns to float and round to 5 decimal places
    num_cols = [
        'Precursor m/z']

    new_df[num_cols] = new_df[num_cols].astype(float).round(5)
    new_df = new_df[~new_df['cf'].str.contains(r'\[')]
    
    if remove_PQCs: 
        new_df = new_df.loc[:, ~new_df.columns.str.contains("PQC")]
    
    return new_df

def select_file(title):
    """Opens a file dialog to select a file and returns the file path."""
    file_path = filedialog.askopenfilename(title=title, filetypes=[("CSV files", "*.csv")])
    return file_path

def merge_files():
    """Handles file selection, merging, and saving the output."""
    # Columns to be removed from the RT dataframe
    columns_to_remove = [
        "Average Rt(min)", "Average Mz", "Metabolite name", "Adduct type", "Post curation result",
        "Fill %", "MS/MS assigned", "Reference RT", "Reference m/z", "Formula", "Ontology",
        "INCHIKEY", "SMILES", "Annotation tag (VS1.0)", "RT matched", "m/z matched",
        "MS/MS matched", "Comment", "Manually modified for quantification", 
        "Manually modified for annotation", "Isotope tracking parent ID", 
        "Isotope tracking weight number", "RT similarity", "m/z similarity", "Simple dot product",
        "Weighted dot product", "Reverse dot product", "Matched peaks count", 
        "Matched peaks percentage", "Total score", "S/N average", 
        "Spectrum reference file name", "MS1 isotopic spectrum", "MS/MS spectrum"
    ]
    
    # Prompt user to select the RT file
    rt_file = select_file("Select the MS Dial Retention time CSV file")
    if not rt_file:
        messagebox.showerror("Error", "No MS Dial Retention time selected!")
        return
    
    # Prompt user to select the MS-Dial file
    ms_dial_file = select_file("Select the MS-Dial intensity CSV file")
    if not ms_dial_file:
        messagebox.showerror("Error", "No MS-Dial intensity file selected!")
        return
    
    try:
        # Load the CSV files
        rt_df = pd.read_csv(rt_file, skiprows = 4)
        
        ms_dial_df = pd.read_csv(ms_dial_file, skiprows = 4)
        ms_dial_df = ms_dial_df.drop(columns = ['MS/MS spectrum'])
        # Remove specified columns from the RT dataframe
        rt_df = rt_df.drop(columns=[col for col in columns_to_remove if col in rt_df.columns], errors='ignore')
        
        

        # Merge the DataFrames on 'Alignment ID'
        global merged_df
        merged_df = pd.merge(rt_df, ms_dial_df, on="Alignment ID", how="inner", suffixes=['_RT_sec', '_Abn'])
        global new_df
        new_df = modify_data_frame(merged_df, remove_PQCs= True)
        new_df.loc[:, new_df.columns.str.contains('_RT_sec$')] = new_df.loc[:, new_df.columns.str.contains('_RT_sec$')].mul(60).round(3)
        
        
        new_df['Lipid Unique Identifier'] = new_df['Lipid Unique Identifier'].apply(lambda x: strip_ms_flags(x))
        
        new_df = deduplicate_best_ms_flag(new_df)
        # Save the merged DataFrame
        save_path = filedialog.asksaveasfilename(
            title="Save DeuteRater-ready File",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if save_path:

            new_df.to_csv(save_path)
            
            messagebox.showinfo("Success", f"Merged file saved to: {save_path}")
        else:
            messagebox.showinfo("Cancelled", "Save operation was cancelled.")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{e}")

# Create the main application window
root = tk.Tk()
root.title("CSV Merger")

# Create a button to initiate the merge process
merge_button = tk.Button(root, text="Merge CSV Files", command=merge_files)
merge_button.pack(pady=20)

# Run the application
root.mainloop()
