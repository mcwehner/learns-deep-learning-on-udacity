## Cross entropy

### Rough notes

Basic formulae:

| Name	| [AsciiMath][]	|
|-	|-	|
| Entropy	| `H(p) = sum_x p(x)  log(1 / (p(x)))`	|
| Cross entropy	| `H_p(q) = sum_x q(x)  log(1 / (p(x)))`	|
| KL divergence	| `D_q(p) = H_q(p) - H(p)`	|
| Loss<sup>1</sup>	| `cc"L" = 1/N sum_i D(S(Wx_i+b, L_i)`	|

1. _Loss_ is defined here as the average cross entropy.

[AsciiMath]: http://asciimath.org


### Resources

* [Visual Information Theory](http://colah.github.io/posts/2015-09-Visual-Information/)
* [Information geometry](https://en.wikipedia.org/wiki/Information_geometry)
