# Bounded Space Differentially Private Quantiles

Estimating the quantiles of a large dataset is a fundamental problem in both the
streaming algorithms literature and the differential privacy literature. However,
all existing private mechanisms for distribution-independent quantile computation require space at
least linear in the input size. In this work, we devise a differentially private algorithm for
the quantile estimation problem with strongly sublinear space complexity.
Our approach builds upon deterministic streaming algorithms
for non-private quantile estimation 
instantiating the exponential mechanism using a utility function defined on sketch items,
while (privately)
sampling from intervals defined by the sketch. We also present
another algorithm based on histograms that is especially
suited to the multiple quantiles case.
We implement our algorithms
and experimentally evaluate them on synthetic and real-world datasets. 

Install required packages via
```
pip3 install -r requirements.txt
```

To generate some test data, use
```
python3 Datagen.py
```

To run experiments with a small approximation parameter, use
```
python3 Experiments.py small
```

The previous command saves results to a database file
(e.g., `Small_approx_error` and `Small_approx_sizes`).

Then using
```
python3 Tables.py small
```
you can print out the results in a tabular format that can be readily
displayed or plotted.

