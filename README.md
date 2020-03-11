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
python gbfit.py --seed 1792656085 --Tobs 6.28e7 --injfile testinj.dat --f0 1e-3 --fdot 0 --fddot 0 --outfile testinj.nc
```

And then have a look at, and regenerate, the plots in `testinj.ipynb`.

A more challenging test (much larger `fdot`) will be

```shell
python gbfit.py --seed 1300839501 --Tobs 6.28e7 --injfile testinj2.dat --f0 0.015248 --fdot 2.58342e-14 --fddot 1.60490223549e-25 --outfile testinj2.nc
```

(Plots in `testinj2.ipynb`.)  You can also try running this system for more or
less observation time.
