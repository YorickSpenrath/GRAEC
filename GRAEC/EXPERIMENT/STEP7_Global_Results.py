"""
This script combines the results of all classifier combination approaches and wraps them into a graph
"""

from GRAEC.Functions import Filefunctions
from GRAEC.AUXILIARY_FUNCTIONS import Name_functions
from GRAEC import Parameters
import matplotlib.pyplot as plt

from GRAEC.EXPERIMENT import STEP8_Over_Time_Scoring


def run():
    for metric in ['accuracy', 'f1']:
        best_graec_score = -1
        best_graec_parameters = None
        fig, ax = plt.subplots()
        y = []
        x_labels = []
        colours = []

        with open(Name_functions.metric_table(metric), 'w+') as wf:

            wf.write('\\begin{table}[]\n')
            wf.write('\\centering\n')
            wf.write('\\begin{tabular}{|c|c|c|c|}\n')
            wf.write('\\hline\n')
            wf.write('$S$ & $\\beta$ & $\\tau$ & {}\\\\\n'.format(Parameters.Tex_dict[metric]))
            wf.write('\\hline\n')

            for S in Parameters.S_values:
                names = ['N{}', 'R{}', 'GR{}']
                fn_scores = [Name_functions.S_naive_score(S, metric),
                             Name_functions.S_recent_score(S, metric),
                             Name_functions.S_GRAEC_score(S, metric)]
                colour_values = ['r', 'orange', 'b']

                for i in range(3):
                    with open(fn_scores[i], 'r') as rf:
                        if i == 2:
                            # Our solution
                            (B, T, P) = rf.readline()[:-1].split(';')[0:4]
                            score = float(rf.readline()[:-1])
                            wf.write('{}&{}&{}&{:.3f}\\\\\n'.format(S, B, T, score))
                            if score > best_graec_score:
                                best_graec_score = score
                                best_graec_parameters = '{};{};{};{}\n'.format(S, B, T, P)
                        else:
                            score = float(rf.readline()[:-1])
                        x_labels.append(names[i].format(S))
                        y.append(score)
                        colours.append(colour_values[i])

            wf.write('\\hline\n')
            wf.write('\\end{tabular}\n')
            wf.write('\\caption{{Optimal values for $S$, $\\beta$, and $\\tau$, and their {} scores}}\n'.format(
                Parameters.Tex_dict[metric]))

            wf.write('\\label{}\n')
            wf.write('\\end{table}\n')

        ax.bar(range(len(x_labels)), y, color=colours)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Method')
        ax.set_ylim(0, max(y) + 0.05)
        ax.set_ylabel('{} score'.format(metric))
        ax.set_title('{} scores for different concept drift solutions'.format(metric))
        for (xp, yp) in zip(range(len(x_labels)), y):
            ax.text(xp - 0.36, yp, '{:.2f}'.format(yp))

        # Save graph to disc
        fn = Name_functions.metric_figure(metric)
        fig.set_size_inches(20 / 2.56, 10 / 2.56)
        plt.savefig(fn, bbox_inches='tight')
        plt.close()

        # Save best graec parameters to disc
        fn = Name_functions.best_graec()
        with open(fn, 'w+') as wf:
            wf.write(best_graec_parameters)


def run_on():
    run()
    STEP8_Over_Time_Scoring.run_on()


def restart():
    undo()
    run_on()


def undo():
    Filefunctions.delete(Name_functions.results_folder())
    STEP8_Over_Time_Scoring.undo()


if __name__ == '__main__':
    run()
