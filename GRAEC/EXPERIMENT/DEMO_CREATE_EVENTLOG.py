"""
Creates a Demo event log from scratch
"""
from GRAEC.AUXILIARY_FUNCTIONS import Name_functions
from GRAEC import Parameters
from GRAEC.AUXILIARY_FUNCTIONS.Concept_Drifters import BaseConceptDrifter
import numpy as np

# repeatability
np.random.seed(684)

activities = Parameters.activity_labels
topics = Parameters.activity_labels
number_activities = len(topics)
shape = 20 / number_activities
scale = 14 / 20
long_time_factor = number_activities + 1

dimension_size = Parameters.dimension_size
number_of_months = Parameters.number_of_months
cases_per_month = Parameters.cases_per_month_per_topic * number_activities


class ConceptDriftedEventLogCreator:

    # Time distributions

    def __init__(self, concept_drifter=Parameters.concept_drifter):
        assert (isinstance(concept_drifter, BaseConceptDrifter))
        self.ConceptDrifter = concept_drifter

    @staticmethod
    def activity_duration(short_time):
        assert (isinstance(short_time, bool))
        if short_time:
            return np.random.gamma(shape=shape, scale=scale) / 30
        else:
            return np.random.gamma(shape=shape * long_time_factor, scale=scale) / 30

    def generate_case(self, month, case_number):
        assert (isinstance(case_number, int))
        assert (isinstance(month, int))

        index = case_number % number_activities
        label_switch = (month % 12) // (12 / number_activities)

        time = month + np.random.rand()
        topic = topics[index]
        pages = np.random.randint(dimension_size)
        publications = np.random.randint(dimension_size)
        short_case = self.ConceptDrifter.is_short(time=time, pages=pages, publications=publications)

        case_id = '{}_{}_{}_{}'.format(month, case_number, str(short_case)[0], label_switch)

        c_info = '{};{};{};{}\n'.format(case_id, topic, pages, publications)
        e_info = '{};{};{}\n'.format(case_id, time, 'Start_Case')

        if short_case:
            for j in range(number_activities):
                time += self.activity_duration(True)
                e_info += '{};{};{}\n'.format(case_id, time, activities[j])
        else:
            for j in range(number_activities):
                time += self.activity_duration(((index + label_switch) % number_activities) != j)
                e_info += '{};{};{}\n'.format(case_id, time, activities[j])
        return c_info, e_info

    def run(self, fn_cases, fn_event_log):
        with open(fn_cases, 'w+') as wf_cases:
            # Info about each column:
            wf_cases.write('Case_id;Topic;Pages;Publications\n')
            wf_cases.write('-;FALSE;TRUE;TRUE\n')
            wf_cases.write('-;TRUE;FALSE;FALSE\n')
            wf_cases.write('ID;X;X;X\n')
            with open(fn_event_log, 'w+') as wf_event_log:
                for month in range(number_of_months):
                    for c in range(cases_per_month):
                        c_info, e_info = self.generate_case(month, c)
                        wf_cases.write(c_info)
                        wf_event_log.write(e_info)


def run():
    assert (len(topics) == len(activities))
    assert (number_activities in Parameters.S_values)

    concept_drifter = Parameters.concept_drifter
    ConceptDriftedEventLogCreator(concept_drifter).run(fn_cases=Name_functions.cases_info(),
                                                       fn_event_log=Name_functions.event_log())


if __name__ == '__main__':
    run()
