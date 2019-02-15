# GRAEC
Gradual and Recurrent concept drift Adapting Ensemble Classifier

This is the github to accompany the DSI4 (EDBT 2019) paper on 'Ensemble-Based Prediction of Business Processes Bottlenecks With Recurrent Concept Drifts' by Yorick Spenrath and Marwan Hassani.

Two files are of main importance: SCRIPT.py and Parameters.py

To run the simulation, it is required to change the root_location in Parameters.py. All data will be saved to the root location, under default settings, the simulation will generate about 290 MB of data, which includes ML Models, and text files with the predictions they make. To run the simulation, import the project and run SCRIPT.py, this has been tested in PyCharm Community edition.

Parameters.py contains all settings for the run. It has information on how the simulated event log is build, how concept drift is applied, what types of models are trained, when training starts, and what and how many datapoints are used for assessment.

A complete simulation under the given parameters takes about 45 minutes, tested on a laptop with 16 GB Ram, 2.6Ghz i7 CPU. I have not done an extensive time complexity analysis, but the total running time is more than linear in the duration of the simulation (in the default set to 60 months), since each model predicts all datapoints with a timestamp after the model.

To run the program on your own dataset, set the variable 'Demo' in Parameters.py to False. This requires you to format an event log and information about cases as follows:

In the root folder (as set in Parameters.py) the event_log needs to be named 'Event_Log.csv'. Each line of this file contains one event, with the case ID, timestamp (as float) and activity label, separated by a ';' (no trailing ';', so a total of 2 ';' per line). The file may not contain a header. The case ID and activity label are interpreted as String. Furthermore, a file named 'Cases_info.csv' needs to be included in the root folder. This file needs to have four header lines.

The first header line is interpreted as strings, which are the feature names.
The second header line contains whether the feature should be interpreted as a float.
The third header line contains whether the feature is categorical. The real dataset only contained categorical data, the k-medoid reduction assumes only categorical data (it uses jaccard distance). The simulation dataset is balanced, so no kmedoid reduction is required.
The fourth line contains information of the type of the column. Prediction features should be labeled as 'X', the Case_id should be labeled as 'ID'. There should be exactly one column marked as 'ID'. The script will later create a column with labels, 'Y', and a column with data timestamps 'SPLIT'.

If you have any issues or ideas for improvement, you may contact me at y dot spenrath at student dot tue dot nl, though I am currently writing my master thesis, so it might take a while before I respond.
