# Efficient and Consistent Adversarial Bipartite Matching
This repository is a code example of the marginal distribution formulation of the adversarial bipartite matching in the paper: 
[Efficient and Consistent Adversarial Bipartite Matching](http://proceedings.mlr.press/v80/fathony18a.html).

### Abstract

Many important structured prediction problems, including learning to rank items, correspondence-based natural language processing, and multi-object tracking, can be formulated as weighted bipartite matching optimizations. Existing structured prediction approaches have significant drawbacks when applied under the constraints of perfect bipartite matchings. Exponential family probabilistic models, such as the conditional random field (CRF), provide statistical consistency guarantees, but suffer computationally from the need to compute the normalization term of its distribution over matchings, which is a #P-hard matrix permanent computation. In contrast, the structured support vector machine (SSVM) provides computational efficiency, but lacks Fisher consistency, meaning that there are distributions of data for which it cannot learn the optimal matching even under ideal learning conditions (i.e., given the true distribution and selecting from all measurable potential functions). We propose adversarial bipartite matching to avoid both of these limitations. We develop this approach algorithmically, establish its computational efficiency and Fisher consistency properties, and apply it to matching problems that demonstrate its empirical benefits. 

# Setup

The source code is written in MATLAB 2016b.

### Dependency
The code depends on the followong tools which are also included in the code:

1. [minConf](https://www.cs.ubc.ca/~schmidtm/Software/minConf.html) : a MATLAB tool for optimization of differentiable real-valued multivariate functions subject to simple constraints on the parameters.
2. [munkres](https://www.mathworks.com/matlabcentral/fileexchange/20328-munkres-assignment-algorithm) : Munkres (Hungarian) algorithm to solve assignment problem in polynomial time.

### Dataset

The datasets are stored in `data` folder. The datasets contain extracted feature from the original 2DMOT2015 task in the [Multiple Object Tracking Benchmark](https://motchallenge.net/).


### Experiments

Two files are provided for running adversarial bipartite matching experiment and Structured SVM experiment: 

* `experiment_2ds.m` :
run the adversarial bipartite experiment. 

* `ssvm_experiment_2ds.m` :
run the Structured SVM experiment. 

# License

The license of the adversarial bipartite matching code is [MIT license](https://choosealicense.com/licenses/mit/). The license of the munkres tool can be found in `munkres/license.txt`. Please refer to [minConf](https://www.cs.ubc.ca/~schmidtm/Software/minConf.html) website for the information of its license.

The dataset containing extracted features is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License](https://creativecommons.org/licenses/by-nc-sa/3.0/) following the same license of the original [Multiple Object Tracking Benchmark](https://motchallenge.net/) dataset license. Please refer to [Multiple Object Tracking Benchmark](https://motchallenge.net/) website for a more detailed information of the dataset license.

# Citation (BibTeX)
```
@InProceedings{fathony18a,
  title = 	 {Efficient and Consistent Adversarial Bipartite Matching},
  author = 	 {Fathony, Rizal and Behpour, Sima and Zhang, Xinhua and Ziebart, Brian},
  booktitle = 	 {Proceedings of the 35th International Conference on Machine Learning},
  pages = 	 {1456--1465},
  year = 	 {2018},
  editor = 	 {Jennifer Dy and Andreas Krause},
  volume = 	 {80},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Stockholmsmässan, Stockholm Sweden},
  month = 	 {10--15 Jul},
  publisher = 	 {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v80/fathony18a/fathony18a.pdf},
  url = 	 {http://proceedings.mlr.press/v80/fathony18a.html},
}
```
# Acknowledgements 
 This research was supported in part by NSF Grants RI-#1526379 and CAREER-#1652530.