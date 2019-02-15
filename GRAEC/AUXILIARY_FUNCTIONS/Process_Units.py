"""
Process units for handling event logs
"""
from GRAEC.Functions import Filefunctions


class Event:

    def __init__(self, case_id, time, act):
        self.case_id = case_id
        assert (isinstance(time, float) or isinstance(time, int))
        self.time = time
        self.act = act


class Case:

    def __init__(self, case_id):
        self.case_id = case_id
        self.events = []

    def add_event(self, event):
        assert (isinstance(event, Event))
        assert (event.case_id == self.case_id)
        self.events.append(event)
        return event

    def get_end(self):
        return max(self.events, key=lambda e: e.time).time

    def get_start(self):
        return min(self.events, key=lambda e: e.time).time

    def get_duration(self):
        return self.get_end() - self.get_start()

    def get_trace(self):
        return [i.act for i in self.get_sorted_events()]

    def get_sorted_events(self):
        self.events.sort(key=lambda e: e.time)
        return self.events

    def str_trace(self):
        return ';'.join(self.get_trace())

    def get_event(self, k):
        return self.get_sorted_events()[k]


class EventLog:

    def __init__(self, filename):
        assert Filefunctions.exists(filename)
        # We implement the event log as a dict, this allows easier reference when adding an event to the case
        # Since each event has a reference to a case_id, not to a case itself
        self.cases = dict()
        with open(filename, 'r') as rf:
            for line in rf:
                # for each line
                case_id, timestamp, act = line[:-1].split(';')
                # create an event
                e = Event(case_id=case_id, time=float(timestamp), act=act)
                # add it to the corresponding case (or create the new case if necessary)
                self.cases.setdefault(case_id, Case(case_id=case_id)).add_event(e)

    def get_case(self, case_id):
        return self.cases.get(case_id, None)

    def get_splits(self, s):
        ret = dict()
        for c in self.cases.values():
            start_time = c.get_start()
            period = int(start_time / s)
            ret.setdefault(period, set()).update({c})

        return ret
