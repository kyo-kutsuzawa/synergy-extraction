# Extraction of time-varying synergies

## Summary

![Definition of time-varying synergies](https://latex.codecogs.com/svg.image?x[t]\approx\sum_{i=1}^{N}c_iw_{J(i)}[t-t_i])

where x is a time-series, w is a synergy, ci and ti indicate the amplitude and onset time at i-th activation.
J(i) indicates the index of synergy at i-th activation.

## Usage

The extraction code is implemented in `timevarying.py`.
Briefly, `timevarying.extract()` can be used for synergy extraction.

An example is implemented in `example.py`.
You can also implement extraction codes for your dataset by following `example.py` and `timevarying.extract()`.
