## Normalized Inputs and Initial Weights

### Rough Notes

#### Zero mean, equal variance

Wherever possible, our variables should have zero mean, and equal variance:

* Mean:	`X_i = 0`
* Variance:	`sigma(X_i) = sigma(X_j)`

#### Conditioning

A _badly conditioned_ problem is one where the optimizer has to do a lot of searching in order to find a good solution. The opposite of this is a _well conditioned_ problem.

#### Images


`((R-128)/128), ((G-128)/128), ((B-128)/128)`
