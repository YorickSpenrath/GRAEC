from GRAEC import Parameters
from GRAEC.EXPERIMENT import STEP4_Train_Test_Split, STEP7_Global_Results, STEP8_Over_Time_Scoring, \
    STEP5_Predictions, STEP3_Probability_Predictions, STEP6_Global_Scoring, DEMO_CREATE_EVENTLOG, \
    STEP1_Reducing_Class_Sizes, STEP9_Over_Time_Results, STEP0_Split_And_Label, STEP2_Model_Training
from GRAEC.Functions import Filefunctions


# This script runs all other scripts in the project. MS stands for Moment / S, since experiments differ in the moment in
# in the process when the prediction takes place, and the subset period length S

def print_state(s):
    print('-' * 25)
    print(s)
    print('-' * 25)


if Parameters.Demo:
    print('Running Demo')
    print_state('Building Dataset')
    # Remove old folder
    Filefunctions.delete(Parameters.root_location)
    # Create new folder
    Filefunctions.make_directory(Parameters.root_location)
    DEMO_CREATE_EVENTLOG.run()
else:
    print('Running Real Dataset')

print_state('Splitting and labeling event log')
STEP0_Split_And_Label.run()

# Reduce the size of the data set if needed
print_state('Reducing Data Size')
STEP1_Reducing_Class_Sizes.run()

# Train all models
print_state('Training')
STEP2_Model_Training.run()

# Predict all test points, saving the predictions (this will save time recalculating all probabilities later)
print_state('Extracting Probabilities')
STEP3_Probability_Predictions.run()

# Split all the test data into GRAEC training and GRAEC testing points. The GRAEC training points will be used to
# optimise the parameters of the proposed solution (GRAEC = Beta, Epsilon, P, Tau)
print_state('Splitting')
STEP4_Train_Test_Split.run()

# From the predicted probabilities, find the predicted labels
print_state('GRAEC Optimisation and Assessment')
STEP5_Predictions.run()

# Find the score for each method
print_state('Scoring')
STEP6_Global_Scoring.run()

# Parse the results into a figure
print_state('Resulting')
STEP7_Global_Results.run()

# Calculate the metrics over time
print_state('Assessing metric over time')
STEP8_Over_Time_Scoring.run()

# Create plots over time for different parameter settings
print_state('Creating plots of metrics over time')
STEP9_Over_Time_Results.run()
