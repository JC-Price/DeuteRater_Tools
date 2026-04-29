At this point you should have some data that was collected by the LGR
instrument! Grab the data off of the instrument if you havn't already 
and then tranfer the data to a safe place. At this point we are really
only interested in the .csv file with a name similar to 
"h2o_20230819_007_LIMS.csv", so make sure you have that on your computer. 

Go to \Price_lab_CRDS_workflow_programs\LGR_data_processor_5.0_executable\dist
and click on LGR_data_processor_gui_5.0.exe. After a few seconds the program 
should open up a graphical user interface with some places to enter 
information. Give your analysis a snappy title and then click "Choose 
Input File", at this point direct to the .csv discussed above. Next, 
chose an output location for your analysis. It will generate a folder 
in that location with all of the analysis information. Then, enter in
the D2O concentrations of your standard curve (just integers or decimal 
values with leading zeros) and make sure that everything is comma-delimited
(WITHOUT SPACES IN-BETWEEN). Just follow the preloaded format if you have 
any questions :) Then enter in your Identifiers and Subidentifiers that you
designated in your LGR sample .csv. These are case-sensitive and should also
be comma-delimited. 

After that you are good to press "Process Data". A folder with time and enrichment
graphs, your standard curve and dataframes containing the calculuated D2O percentages
and standard deviations for each of your samples and standards will also be included. 
There will also be a folder called asymptote calculation graphs which will show the 
D/H for each sample injection (each injection is simply diluting what was previously in
the LGR cavity, so the measurements follow an expontial decay equation with predictable 
assymptotes, which represent the true D/H value for each sample.)

