# GBPyro

Galactic binaries in LISA with Pyro.  Note: I was having trouble with
[Pyro](https://pyro.ai), and ended up switching to [PyMC3](https://docs.pymc.io)
for the sampling for this project.  Turned out the problems were mine, but
sticking with the working code; so the repo name is unfortunate.

If you want to see how this code was developed, have a look at the
`LISASprint.ipynb` notebook, which was the initial testing / developing of the
code.

Try out the command-line:

```shell
python gbfit.py --seed 1792656085 --Tobs 6.28e7 --injfile testinj.dat --f0 1e-3 --sigma-f0 1.6e-8 --outfile testinj.nc
```

And then have a look at the plots in `testinj.ipynb`.

If you want an actual test on a verification binary, have a go at
