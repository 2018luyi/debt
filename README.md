# Double empirical Bayes testing (DEBT)

This repository implements the paper:

```
Double Empirical Bayes Testing
W. Tansey, Y. Wang, R. Rabadan, D. M. Blei
International Statistical Review, 2020.
[arxiv link here]
```

The paper lays out (and this repository implements) an empirical Bayes approach to analyzing multi-experiment studies with side information for each experiment. The running example is cancer cell line drug screenings. In these experiments, the scientist wishes to test both whether the drug had any effect (we call this Stage 1) and then wishes to select the molecular drivers of response (we call this Stage 2). Both stages of selection are posed as multiple hypothesis testing problems where we use neural networks and an empirical Bayes knockoffs procedure to select significant results with control over the false discovery rate.

If you use this code, please cite the above paper.

## Requirements

This code uses `python3`, `pytorch`, `statsmodels`, and the usual suspects: `numpy`, `scipy`, `matplotlib`, `seaborn`, etc.

## Synthetic experiments
To recreate the synthetic benchmarks, simply run `python python/synthetic.py`.

This will save data to `data/pure-synthetic/`. To recreate the plots from the paper, run `python python/plot_results.py pure-synthetic`. They will be saved to the `plots/` directory.

## Cancer dataset experiments
To recreate the cancer dataset experiments, first download the cancer dataset files [here](https://www.dropbox.com/sh/f745ejqacnhu4ub/AACrF6IqKNH6ff-6NYnoVoYAa?dl=0) and save them all to the folder `data/cancer/`. Then run the command `python python/cancer.py --save`.

This will save data to `data/cancer/`. To recreate the plots for the dataset experiments in the paper, run `python python/plot_results.py cancer`. They will be saved to the `plots/` directory.

Note that this will only recreate the results for the overall dataset. The Nutlin-3a results were from a different random seed and may be slightly different than in the paper. It's annoying to not be able to provide an exact recreation of this, but this variability is actually a good example of the broader issue with using any one-shot knockoff procedure.

