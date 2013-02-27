gEpiCount

Genome Wide Scanning of binary Boolean epistatic interactions.
GPU accelerated.
Class information entropy based significance evaluation.

Copyright C.Riveros, 2012.

Test files provided, data/hap1.xxx, are taken from plink demo data,
see: http://pngu.mgh.harvard.edu/~purcell/plink/tutorial.shtml
data/thinhap1.xxx is derived from previous file by randomly selecting 1020 SNPs.

R's implementation of the hypergeometric distribution has served as inspiration
for the trimmed down version used in Fisher's exact test.

Compilation:
------------

gEpiCount can be compiled as a multiple archive CUDA project.
Requires OpenMP and thrust:: libraries.
Requires compute capability >= 20

Basic usage:
------------

gEpiCount -i data/thinhap1 -N 100

If invoked without arguments it will produce a short help of options.


Contact Info:
-------------

Questions and other matters please reach me at: 
carlos dot riveros at newcastle dot edu dot au.