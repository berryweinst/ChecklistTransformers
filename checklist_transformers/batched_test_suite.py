import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from checklist.test_types import MFT
from checklist.test_suite import TestSuite

class BatchedTestSuite(TestSuite):

    def __init__(self, seed=123, batch_size=1, **kwargs):
        self.format_example_fn = kwargs['format_example_fn']
        self.print_fn = kwargs['print_fn']
        self.tests = kwargs['tests']
        self.info = kwargs['info']
        self.test_ranges = kwargs['test_ranges']
        self.seed = seed
        self.batch_size = batch_size


    def run(self, predict_and_confidence_fn, verbose=True, **kwargs):
        """Runs all tests in the suite
        See run in abstract_test.py .

        Parameters
        ----------
        predict_and_confidence_fn : function
            Takes as input a list of examples
            Outputs a tuple (predictions, confidences)
        overwrite : bool
            If False, raise exception if results already exist
        verbose : bool
            If True, print extra information
        n : int
            If not None, number of samples to draw
        seed : int
            Seed to use if n is not None

        """
        for n, t in self.tests.items():
            if verbose:
                print('Running', n)
            t.run(predict_and_confidence_fn, verbose=verbose, seed=self.seed, batch_size=self.batch_size, **kwargs)


class BatchedMFT(MFT):
    def __init__(self, **kwargs):
        self.data = kwargs['data']
        self.expect = kwargs['expect']
        self.labels = kwargs['labels']
        self.meta = kwargs['meta']
        self.agg_fn = kwargs['agg_fn']
        self.templates = kwargs['templates']
        self.name = kwargs['name']
        self.capability = kwargs['capability']
        self.description  = kwargs['description']
        self.print_first = kwargs['print_first']
        self.run_idxs = kwargs['run_idxs']
        self.result_indexes = kwargs['result_indexes']
        self.results = kwargs['results']

    def run(self, predict_and_confidence_fn, overwrite=False, verbose=True, n=None, seed=None, batch_size=1):
        """Runs test

        Parameters
        ----------
        predict_and_confidence_fn : function
            Takes as input a list of examples
            Outputs a tuple (predictions, confidences)
        overwrite : bool
            If False, raise exception if results already exist
        verbose : bool
            If True, print extra information
        n : int
            If not None, number of samples to draw
        seed : int
            Seed to use if n is not None

        """
        # Checking just to avoid predicting in vain, will be created in run_from_preds_confs
        self._check_create_results(overwrite, check_only=True)
        examples, result_indexes = self.batched_example_list_and_indices(n, seed=seed, batch_size=batch_size)

        if verbose:
            print('Predicting %d examples' % len(examples[0]))
        preds_dict = predict_and_confidence_fn(examples)
        task_group_size = len(self.labels[0])
        num_layers = len(preds_dict.keys())

        dfs_error = [pd.DataFrame(columns=['ts', 'neu', 'value']) for _ in range(num_layers)]
        dfs_conf = [pd.DataFrame(columns=['ts', 'neu', 'value']) for _ in range(num_layers)]

        for layer in range(num_layers):
            for ts in preds_dict[layer].keys():
                for neu in preds_dict[layer][ts].keys():
                    preds = preds_dict[layer][ts][neu]
                    answers = [p['answer'] for p in preds]
                    answers = [answers[x:x+task_group_size] for x in range(0, len(answers), task_group_size)]
                    error = 1. - sum([p==r for p,r in zip(answers, self.labels)]) / len(self.labels)
                    dfs_error[layer] = dfs_error[layer].append({'ts': ts, 'neu': neu, 'value': error}, ignore_index=True)

                    scores = [p['score'] for p in preds]
                    conf = sum(scores) / len(scores)
                    dfs_conf[layer] = dfs_conf[layer].append({'ts': ts, 'neu': neu, 'value': conf}, ignore_index=True)

        self.plot_heatmaps(dfs_error, 'Per neuron error', num_layers)
        self.plot_heatmaps(dfs_conf, 'Per neuron confidence by score', num_layers)



    def plot_heatmaps(self, dfs, title, num_layers):

        nrows = num_layers//3
        ncols = num_layers//2
        nrows += num_layers - nrows*ncols

        vmin = 1
        vmax = 0
        for df in dfs:
            vmin = min(df['value']) if vmin > min(df['value']) else vmin
            vmax = max(df['value']) if vmax < max(df['value']) else vmax

        fig = plt.figure()
        fig.suptitle(title, fontsize=15)
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(nrows, ncols),
                        axes_pad=0.3,
                        share_all=True,
                        label_mode="L",
                        cbar_location="right",
                        cbar_mode="single",
                        )
        for i, (df, ax) in enumerate(zip(dfs, grid)):
            im = ax.imshow(df.pivot(index='ts', columns='neu', values='value'), vmin=vmin, vmax=vmax)
            ax.set_title('layer %d' % (i))

        grid.cbar_axes[0].colorbar(im)
        for cax in grid.cbar_axes:
            cax.toggle_label(False)

        plt.show()


    def batched_example_list_and_indices(self, n=None, seed=None, batch_size=1):
        """Subsamples test cases

        Parameters
        ----------
        n : int
            Number of testcases to sample
        seed : int
            Seed to use

        Returns
        -------
        tuple(list, list)
            First list is a list of examples
            Second list maps examples to testcases.

        For example, let's say we have two testcases: [a, b, c] and [d, e].
        The first list will be [a, b, c, d, e]
        the second list will be [0, 0, 0, 1, 1]

        Also updates self.run_idxs if n is not None to indicate which testcases
        were run. Also updates self.result_indexes with the second list.

        """
        if seed is not None:
            np.random.seed(seed)
        self.run_idxs = None
        idxs = list(range(len(self.data)))
        if n is not None:
            idxs = np.random.choice(idxs, min(n, len(idxs)), replace=False)
            self.run_idxs = idxs
        if type(self.data[0]) in [list, np.array]:
            all = [(i, y) for i in idxs for y in self.data[i]]
            result_indexes, examples = map(list, list(zip(*all)))
        else:
            examples = [self.data[i] for i in idxs]
            result_indexes = idxs# list(range(len(self.data)))
        self.result_indexes = result_indexes
        examples = tuple(zip(*examples))
        return examples, result_indexes