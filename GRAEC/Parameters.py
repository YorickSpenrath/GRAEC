"""
Process Parameter settings: global settings for experiments
"""

from GRAEC.Functions import Model_Functions
from GRAEC.AUXILIARY_FUNCTIONS.Concept_Drifters import LinearConceptDrifter, LinearTimer

Demo = True

# root location of a run
root_location = 'C:/Users/Yorick Spenrath/Desktop/DSI4_Data'
assert(root_location != '')

# Algorithm values
S_values = [1, 2, 3, 4] if Demo else [7, 14, 21, 28]
d = 21.0 / 30.0 if Demo else 21.0
alpha = 0.2
GRAEC_beta = list(range(0, 3))
GRAEC_p = [12 if Demo else 365.25]
GRAEC_tau = [0] + [10 ** i for i in range(-2, 3)]

# Model values
# Naive model is trained with all data with timestamp up to including this
train_time_naive_stop = 12 if Demo else 457
# The testing of all methods is based on data with timestamp bigger than this
test_time_start = 48 if Demo else 622

# This part of the test data is used for assessment
assessment_test_split = 0.20

# Alternatively, one could chose to take the chronologically last part of the test data for assessment
take_test_split_chronological = False

# Model Parameters
# The size of largest class may at most be {LargeSmallFactor} times the size of the smallest class.
LargeSmallFactor = 2
# Force a gridsearch over the parameters of each model if {force_gridSearch} is True
force_gridSearch = False
# At most {max_iter} hyperparameter combintions are used for each ML model
max_iter = 20
# models are trained with a {cv} stratified cross-fold validation
cv = 5

# Use the following models for training
# See Model_Functions for more information
used_models = Model_Functions.get_Models("fast_classifiers" if Demo else "fast_classfiers")

# Simulation Parameters
# The number of activities a single process has. The number of topics is the same
number_of_activities_in_simulation = 4
# The number of months in the simulation
number_of_months = 60
# The number of cases each topic has in each month
cases_per_month_per_topic = 50
# The maximum number of pages and publications a case has
dimension_size = 100
# Class to handle the gradual concept drift
concept_drifter = LinearConceptDrifter(number_activities=number_of_activities_in_simulation,
                                       number_of_months=number_of_months,
                                       dimension_size=dimension_size,
                                       timer=LinearTimer(t_e=number_of_months))

# Process Label Parameters
# labels in the simulation
simulation_activity_labels = [i for (j, i) in enumerate('ABCDEFGHIJKL') if j < number_of_activities_in_simulation]
# labels in the real dataset (masked for confidality)
real_activity_labels = ['Bottleneck_80', 'Bottleneck_82', 'Bottleneck_85', 'Bottleneck_90', 'Bottleneck_UNKOWN']
# activity labels used in a run
activity_labels = simulation_activity_labels if Demo else real_activity_labels
# label name for short cases
short_label = 'Short_Case'
# all label names
all_labels = [short_label] + activity_labels

# Tex conversion
Tex_dict = {'Beta': r'$\beta$',
            'Tau': r'$\tau$',
            'P': r'$P$',
            'S': r'$S$',
            'f1': r'$F_1$',
            'accuracy': 'ACC',
            }

# Pandas import of over-time assessment
parameter_evaluation_dtypes = {'Beta': float,
                               'P': float,
                               'Tau': float,
                               'S': int,
                               'f1': float,
                               'accuracy': float,
                               'NumEntries': int,
                               'Day': int,
                               }
