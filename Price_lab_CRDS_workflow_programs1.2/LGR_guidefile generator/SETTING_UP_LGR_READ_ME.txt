This program is designed to make your experience with the 
Cavity Ring Down instrument (LGR) an easy one, preventing hours and hours
of needless clicking and manual randomizing. 
-Coleman Nielsen

1. The first thing you will want to do is open up a copy of the 
LGR_standard_curve_template.xlsx (one should be included in this folder).
In short, there are four columns, "Name", "S/N", "Tray" and "Position". 

Name column:
It is best not to change the content
of this column. Make a standard curve of 7 D2O dilutions that will cover 
the expected range for your measurements (the SOP for how to best use this 
should also be included in the root of the Price_lab_CRDS_workflow_programs
zipped folder. See: \Price_lab_CRDS_workflow_programs\Preparing_LGR_samples_SOP.pdf). 

S/N column:
Place the percentages of your enrichment here (any format you would like!)

Tray & Position columns:
You should not need to change these either. It is the current practice to 
place all of the standards from position 34 to positon 54 in the fourth sample
tray in the auto loader. This ensures there is ample space for the samples.

Finally, save the modified .xlsx (excel) file as a .csv with a name that makes 
sense.

2. The next thing you should do is open up LGR_sample_template.xlsx. In short, there are also
four columns, "Name", "S/N", "Tray", and "Position" (but these are handled a little
bit differently than before, and unfortunately you have a bit more to enter in). Make
sure to make use of excel's "drag feature" as many of your entries are going to have
similar names and everything from here on out is case-sensative).

Name column: Begining with the end in mind, let's talk about 
LGR_data_processor_gui_5.0.exe, the other program you will use after the LGR measures
your samples. LGR_data_processor_gui_5.0.exe uses a simple multiplicative logic to 
determine how many different time-versus-enrichment graphs it generates. The total number
of generated graphs will be the number of "identifiers" multiplied by the number of 
"subidentifiers" that you establish in the names of your samples. It is designed this way
due to the fact that oftentimes two different categorical variables are evaluated simultaneously
during a biological experiment. The "identifiers", and "subidentifiers" 
represent the variants of those variables. When you are doing your experiment, you can
choose to use both identifiers and subidentifiers, just identifiers, or neither. The benifit
of at least using identifiers is that your time and enrichment graphs will have a 
descriptive title. 

The format of each of your sample names should be as such: 
"identifier_subidentifier_D*_tech**", or "identifier_D*_tech**", or "D*_tech**".

* --> This represents the "day" value of your measurement. Or, how many days since the
begining of the time course. This is often how many days since the IP-injection of D2O, 
or how many days since the subject's first sip of D2O. This value can be in decimal
format (so D0.25 for 6 hours since the start of the time course). 

** --> This represents the technical replicate number (usually 1/300
dilution technical reps, as this is probably the most variable part of the SOP). These
are necessary in order to acheieve a trustworthy analysis. You will be able to see the
standard deviation of these (AKA, how good your pipetting skills are) in the
final_samples_output.csv and final_standards_output.csv, following data analysis. 

S/N column:
This is the easiet part. You can literally put anything you would like here. 

Tray & Position columns:
There are 4 trays and 54 positions per tray. That is a maximum of 216 potential positions between the
standards and samples (or a maximum of 195 samples if you maintain the 21 sample system). 

Hopefully this all makes sense in context with the guidefile!

After you have filled out the second spreadsheet, save it as a .csv with a name that makes sense.

3. Next click on LGR_guidefile_generator.exe (it may take a second to load).
(found in "Price_lab_CRDS_workflow_programs\LGR_guidefile generator\dist"). This will pull up a 
graphical user interface (GUI) with a whole bunch of settings. Once you really understand the ins and outs
of the LGR instrument, you might find yourself wanting to change these settings, otherwise leave them as
they are. Check the box that says "included standards", then select "import samples .csv", and select your
samples file, then click "import standards .csv" and select your standards file. 

Then hit "Save Configuration"! At this point you will have created a guide file for the LGR instrument that
tells it which settings to use and gives it the names, locations and order of each of the samples and 
standards. You might be asking yourself, "but what about randomization?". The entire worklist is completely
randomized, with random standards being run every four samples. 

4. Upload the newly generated LGR guidefile onto a USB. Plug it into the back USB port on the LGR instrument
and click "import/export files". It will allow you then to move the generated guidefile into the hard drive
of the LGR instrument. Then click "configure" and select your guidefile. At this point there will be a
couple of errors that pop up. They are not important so ignore them. After that you are good to go! If you
click on "run" you will see the complete run-list. 

5. At this point, if you have cleaned the syringe, parafilmed your samples, loaded them into the sample trays, 
made sure the LGR instrument is clean, has the spectrum aligned properly and has a fresh septum, 
you are good to go! Click start. 

6. Once the run is over make sure to transfer all of the generated files (sort by date and transfer everything
done in the last day) to a USB! Now you are ready to do some data analysis. 

See: 
\Price_lab_CRDS_workflow_programs\LGR_data_processor_5.0_executable\LGR_DATA_ANALYSIS_READ_ME.txt








 