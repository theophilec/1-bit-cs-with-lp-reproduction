# Reproducing results from "One-bit Compressed Sensing by Linear Programming" by Plan & Vershynin

## Reproduction authors

The reproduction is an independent effort by Théodor Lemerle & Théophile Cantelobre to illustrate Plan & Vershynin's paper with experimental results.

## Summary of the approach

Plan & Vershinyn show that the One-bit Compressed Sensing problem can be solved as a linear program.

Theoretically, the approach has two main ingredients:

* notions of _tesselations_ (sections 3 and 4)
* notion of effective sparsity (section 5)

## Experiments 

### Synthetic sparse vectors

Inspired by the experiments in [6].

We generate sparse vectors, where the non-zero components are iid standard Normal.

### MNIST (image space)

We reconstruct MNIST digits directly in image space.

### Natural images (wavelets space)



## Bibliography

[6]: Boufounos, P. T.; Baraniuk, R. G. 1-bit compressive sensing. 
