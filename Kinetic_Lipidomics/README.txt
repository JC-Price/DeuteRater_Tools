Welcome to Kinetic Lipidomics!

FIRST TIME USERS NOTE: 
Long file paths can be a Major issue for windows machines. In order to prevent issues with our installation wizard please put the Lipidomics_Kinetics_Workflow2 into a folder close
to the root of what ever drive it is saved in (for example, Downloads/Kinetic_Lipidomics or D:Kinetic_Lipidomics both work great). 

We need to set up the python environments. There is two ways you can do this:
A) Easiest if Anaconda is on your home system. 
	-Using the Anaconda command prompt navigate to the Lipidomics_Kinetic_Workflow2 folder with "cd".
	-Run the following commands in the Anaconda command prompt:
		-conda env create -f DeuteRaterEnvironment.yml
		-conda pack -n DeuteRaterEnvironment


		-conda env create -f Kinetic_Lipidomics.yml
		-conda pack -n Kinetic_Lipidomics


B) You can retrieve them from our Box server with:
	- https://byu.box.com/s/jfhbbwbt6fiuc6ev71n8rsralx15ut86 (Kinetic_Lipidomics.tar.gz)
& 
	- https://byu.box.com/s/mvuvyljxqprvjpv94u5apvmtoc9nkeao (DeuteRaterEnvironment.tar.gz)


	-Store these within the root of Lipidomics_Kinetic_Workflow2.


Then you will need to set up the python environments by clicking the Kinetic_Lipidomics_Environment_Wizard.cmd before you do anything else. This 
will extract the necessary python environment and ensure that the hard-coded paths used by the various internal python packages are pointing to the correct places. This process will remove 
the .tar.gz zipped files as it unwraps and activates them. Then it will remove itself. 

This is a software package containing the pieces that the Price Lab uses in their Kinetic Lipidomics workflow (and as featured in Nielsen et al. 2026). 
Each subfolder (save for "shared_python" and "dev-Binomial-NL-EXE") contain click and run .cmd (windows command) files that run their associated .py (Python)
programs through the included python distribution and environment (located in "shared_python"). "dev-Binomial-NL-EXE" contains the most recent development of 
DeuteRater, the Price Lab's kinetic workflow, with a new element, empirical deuterium incorporation number (nL) measurement using the bionomial theorem.  

The .cmd file labeled "Lipidomics_Kinetics_Workflow2" is a software that will walk you through the kinetic lipdomics workflow in its entirety, interpreting raw kinetics mass spectrometry
data into results with statistics and visualizations, explaining concepts and our rational for each step along the way. 

This is intended to be run on a windows system, but the .py files for the individual steps will work fine in most cases on Linux and Macintosh systems.

"Kinetic_lipidomics.yml" contains the version of python and environment that we used while building each part of this package, except for DeuteRater, which has its own
versions of both and are incorporated into the included "DeuteRater.exe".


If you are interested in working with or editing the .py files for your own use. The anaconda command (see anaconda.org for more information) 
"conda env create -f Kinetic_lipidomics.yml" will turn Kinetic_lipidomics.yml  into a Microsoft OS anaconda environment that you can use on your own system to mimic the exact 
conditions that we used in the development of this package.

The same goes for DeuteRaterEnvironment.yml, which is the unique DeuteRater-specific environment that we use.

We wish you the best as you attempt to either recreate our results with our publicly-available raw data (include link here), or with data of your own. 

IF you have any questions or comments, Coleman Nielsen, the first author on Nielsen et al. 2026, or the corresponding author, John C. Price can be contacted at nielsen.coleman2@gmail.com or 
jcprice@byu.edu, respectfully. 




