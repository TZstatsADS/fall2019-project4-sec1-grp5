# Project 4: Algorithm implementation and evaluation: Collaborative Filtering

### [Project Description](doc/project4_desc.md)

Term: Fall 2019

+ Team #
+ Projec title: SGD + ALS with Temporal Dynamic Regularization and KNN Post Processing 
+ Team members
	+ Chen, Haofeng hc2962@columbia.edu
	+ Gong, Yuting yg2641@columbia.edu
	+ Song, Mingming ms5710@columbia.edu
	+ Zhang, Jerry jz2966@columbia.edu
	+ Zheng, Kaiyan kz2324@columbia.edu
	
+ Project summary:
Our goal was to evaluate A1 & A3, (SGD & ALS in Factorization Methods dealing with Temporal Dynamics) 

Factorization Methods
Gradient Descent (Nonprobabilistic)
Alternating Least Squares

Regularization Method
Temporal Dynamics

Postprocessing Method
SVD with KNN

	
**Contribution statement**:  
+ Jerry Zhang: Wrote entire SGD algorithm, KNN algorithm, and temporal dynamic algorithm. Helped develop, research, and code relevant equations for all the algorithms.  Led the direction of the project as presenter and group leader. Helped write functions to plot and process data for all aspects of the project (both objective functions). Produced the most code and relevant work for final result and evaluation. 
+ Haofeng Chen: Derived relevant equations for objective functions + dynamics, helped write KNN algorithm
+ Mingming Song: Wrote ALS algorithm with temporal dynamics
+ Yuting Gong: Helped write ALS algorithm with temporal dynamics, preprocessed bias data
+ Kaiyan Zheng: Helped summarize paper, created half the ppt slides

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
