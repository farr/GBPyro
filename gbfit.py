#!/usr/bin/env python
from pylab import *

from argparse import ArgumentParser
import arviz as az
import os
import os.path as op
import pymc3 as pm
import pymc3.math as pmm
import theano
import theano.tensor as tt
import theano.tensor.fft as ttf

theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

# I sure hope L is in m
def Sn(fs, L=5e9):
    omegas = 2*pi*fs
    fstar = 2.99792e8/(2*pi*L)
    Sa = 1e-22/(L*L)
    Ss = 9e-30/(omegas*omegas*omegas*omegas)/(L*L)

    c2 = cos(2.0*fs/fstar)
    c = cos(fs/fstar)

    return 4.0/3.0*(1 - c2)*((2 + c)*Ss + 2*(3 + 2*c + c2)*Sa)

def constellation(ts, e=0.00985, R=1, kappa=0, lam=0):
    L = 5e9*2*sqrt(3)*e*R # m ; R in AU

    alpha = 2*pi*ts/(3600.0*24.0*365.25*R**(3.0/2.0)) + kappa
    alpha = tt.reshape(alpha, (1, alpha.shape[0]))

    betas = tt.as_tensor_variable([kappa, kappa + 2*pi/3, kappa + 4*pi/3])
    betas = tt.reshape(betas, (betas.shape[0], 1))

    xs = R*pmm.cos(alpha) + 0.5*e*R*(pmm.cos(2*alpha - betas) - 3*pmm.cos(betas))
    ys = R*pmm.sin(alpha) + 0.5*e*R*(pmm.sin(2*alpha - betas) - 3*pmm.sin(betas))
    zs = -sqrt(3)*e*R*pmm.cos(alpha - betas)

    return tt.stack((xs, ys, zs), axis=2)

def uvk(nhat):
    phi = tt.arctan2(nhat[1], nhat[0])
    cos_theta = nhat[2]
    sin_theta = pmm.sqrt(1 - cos_theta*cos_theta)

    cphi = pmm.cos(phi)
    sphi = pmm.sin(phi)

    u = tt.as_tensor_variable([cos_theta*cphi, cos_theta*sphi, -sin_theta])
    v = tt.as_tensor_variable([sphi, -cphi, 0.0])
    k = tt.as_tensor_variable([-sin_theta*cphi, -sin_theta*sphi, -cos_theta])

    return tt.stack((u,v,k), dim=0)

def transfer(fs, ts, khat, xyz, L=5e9):
    khat = tt.reshape(khat, (1, 3))

    fstar = 2.99792e8 / (2*pi*L)

    tfs_re = tt.zeros((3,3,ts.shape[0]))
    tfs_im = tt.zeros((3,3,ts.shape[0]))
    for i in range(3):
        for j in range(i+1, 3):
            if not (j == i):
                rij = xyz[j,:,:] - xyz[i,:,:]
                r2 = tt.reshape(pmm.sum(rij*rij, axis=1), (-1, 1))
                rij = rij / pmm.sqrt(r2)

                rk = pmm.sum(rij*khat, axis=1)
                w = fs/(2*fstar)*(1-rk)
                wp = fs/(2*fstar)*(1+rk) # Reverse direction, get a plus
                sc = pmm.sin(w)/w
                scp = pmm.sin(wp)/wp

                tfs_re = tt.set_subtensor(tfs_re[i,j,:], sc*pmm.cos(w))
                tfs_im = tt.set_subtensor(tfs_im[i,j,:], sc*pmm.sin(w))

                tfs_re = tt.set_subtensor(tfs_re[j,i,:], scp*pmm.cos(wp))
                tfs_im = tt.set_subtensor(tfs_im[j,i,:], scp*pmm.sin(wp))

    return tfs_re, tfs_im

def dp_dc(xyz, uvk):
    dps = tt.zeros((3,3,xyz.shape[1]))
    dcs = tt.zeros((3,3,xyz.shape[1]))
    for i in range(3):
        for j in range(i+1,3):
            rij = xyz[j,:,:] - xyz[i,:,:]
            r2 = pmm.sum(rij*rij, axis=1)
            rij = rij / tt.reshape(pmm.sqrt(r2), (-1, 1))

            ru = pmm.sum(rij*uvk[0,:], axis=1)
            rv = pmm.sum(rij*uvk[1,:], axis=1)

            dps = tt.set_subtensor(dps[i,j,:], ru*ru - rv*rv)
            dcs = tt.set_subtensor(dcs[i,j,:], ru*rv + ru*rv)
            dps = tt.set_subtensor(dps[j,i,:], dps[i,j,:])
            dcs = tt.set_subtensor(dcs[j,i,:], dcs[i,j,:])

    return dps, dcs

def Ap_Ac(fs, cos_iota):
    cos_iota2 = cos_iota*cos_iota
    fs23 = (pi*fs/1e-3)**(2.0/3.0)

    return (2.0*fs23*(1+cos_iota2), -4.0*fs23*cos_iota)

def y_slow(ts, f0, fdot, phi0, nhat, cos_iota, psi, fhet=None):
    if fhet is None:
        fhet = f0

    df = f0 - fhet

    uvk_ = uvk(nhat)
    khat = uvk_[2,:]

    xyz = constellation(ts)

    dp, dc = dp_dc(xyz, uvk_)

    cp = pmm.cos(2.0*psi)
    sp = pmm.sin(2.0*psi)

    ys_re = tt.zeros((3, 3, ts.shape[0]))
    ys_im = tt.zeros((3, 3, ts.shape[0]))

    for i in range(3):
        kdotx = 499.0048*tt.tensordot(xyz[i,:,:], khat, axes=1) # 499.0048 = 1 AU / c in seconds
        xi = ts - kdotx
        fs = f0 + fdot*xi

        tfs_re, tfs_im = transfer(fs, ts, khat, xyz)
        Ap, Ac = Ap_Ac(fs, cos_iota)

        phi = pi*fdot*xi*xi + phi0 - 2*pi*f0*kdotx + 2*pi*df*ts
        osc_re = pmm.cos(phi)
        osc_im = pmm.sin(phi)

        for j in range(3):
            if not i == j:
                Fp_re = dp[i,j]*Ap*cp - dc[i,j]*Ap*sp
                Fp_im = -dp[i,j]*Ac*sp - dc[i,j]*Ac*cp

                A_re = (Fp_re*tfs_re[i,j,:] - Fp_im*tfs_im[i,j,:])/4
                A_im = (Fp_im*tfs_re[i,j,:] + Fp_re*tfs_im[i,j,:])/4

                ys_re = tt.set_subtensor(ys_re[i,j,:], A_re*osc_re - A_im*osc_im)
                ys_im = tt.set_subtensor(ys_im[i,j,:], A_im*osc_re + A_re*osc_im)

    return (ys_re, ys_im)

def slow_bw(f0, fdot, Tobs, R=1):
    fm = 1.0/(3600.0*24.0*365.25*R**(3.0/2.0))

    return 2*(4 + 2*pi*f0*R*499.0047)*fm + abs(fdot*Tobs)  # 499.0047 is 1 AU in seconds

def next_pow_two(N):
    i = 1
    while i < N:
        i = i << 1
    return i

def y_fd(t0, Tobs, f0, fdot, phi0, nhat, cos_iota, psi, heterodyne_bin, N):
    fhet = heterodyne_bin/Tobs

    ts = t0 + linspace(0, Tobs, N+1)[:-1]

    ys_re, ys_im = y_slow(ts, f0, fdot, phi0, nhat, cos_iota, psi, fhet=fhet)

    ys_re_rfft = ttf.rfft(tt.reshape(ys_re, (-1,N)))
    ys_im_rfft = ttf.rfft(tt.reshape(ys_im, (-1,N)))

    NN = N//2 + 1

    ys_re_rfft = tt.reshape(ys_re_rfft, (3,3,NN,2))
    ys_im_rfft = tt.reshape(ys_im_rfft, (3,3,NN,2))

    y_fd_re = tt.zeros((3,3,N))
    y_fd_im = tt.zeros((3,3,N))

    y_fd_re = tt.set_subtensor(y_fd_re[:,:,:NN], ys_re_rfft[:,:,:,0] - ys_im_rfft[:,:,:,1])
    y_fd_im = tt.set_subtensor(y_fd_im[:,:,:NN], ys_re_rfft[:,:,:,1] + ys_im_rfft[:,:,:,0])

    y_fd_re = tt.set_subtensor(y_fd_re[:,:,NN:], ys_re_rfft[:,:,-2:0:-1,0] + ys_im_rfft[:,:,-2:0:-1,1])
    y_fd_im = tt.set_subtensor(y_fd_im[:,:,NN:], -ys_re_rfft[:,:,-2:0:-1,1] + ys_im_rfft[:,:,-2:0:-1,0])

    return 0.5*(ts[1]-ts[0])*y_fd_re, 0.5*(ts[1]-ts[0])*y_fd_im

def XYZ_freq(yf_re, yf_im, Tobs, heterodyne_bin, N, L=5e9):
    fstar = 2.99792e8/(2.0*pi*L)

    nf = N
    df = 1.0/Tobs
    dfs = linspace(0, df*nf/2, int(round(nf/2))+1)
    dfs = concatenate((dfs, -dfs[-2:0:-1]))
    fs = heterodyne_bin*df + dfs

    c1 = pmm.cos(-fs/fstar)
    s1 = pmm.sin(-fs/fstar)

    c2 = pmm.cos(-2*fs/fstar)
    s2 = pmm.sin(-2*fs/fstar)

    c3 = pmm.cos(-3*fs/fstar)
    s3 = pmm.sin(-3*fs/fstar)

    X_re = yf_re[0,1,:]*c3 - yf_im[0,1,:]*s3 \
           - (yf_re[0,2,:]*c3 - yf_im[0,2,:]*s3) \
           + yf_re[1,0,:]*c2 - yf_im[1,0,:]*s2 \
           - (yf_re[2,0,:]*c2 - yf_im[2,0,:]*s2) \
           + yf_re[0,2,:]*c1 - yf_im[0,2,:]*s1 \
           - (yf_re[0,1,:]*c1 - yf_im[0,1,:]*s1) \
           + yf_re[2,0,:] - yf_re[1,0,:]

    X_im = yf_im[0,1,:]*c3 + yf_re[0,1,:]*s3 \
           - (yf_im[0,2,:]*c3 + yf_re[0,2,:]*s3) \
           + yf_im[1,0,:]*c2 + yf_re[1,0,:]*s2 \
           - (yf_im[2,0,:]*c2 + yf_re[2,0,:]*s2) \
           + yf_im[0,2,:]*c1 + yf_re[0,2,:]*s1 \
           - (yf_im[0,1,:]*c1 + yf_re[0,1,:]*s1) \
           + yf_im[2,0,:] - yf_im[1,0,:]

    Y_re = yf_re[1,2,:]*c3 - yf_im[1,2,:]*s3 \
           - (yf_re[1,0,:]*c3 - yf_im[1,0,:]*s3) \
           + yf_re[2,1,:]*c2 - yf_im[2,1,:]*s2 \
           - (yf_re[0,1,:]*c2 - yf_im[0,1,:]*s2) \
           + yf_re[1,0,:]*c1 - yf_im[1,0,:]*s1 \
           - (yf_re[1,2,:]*c1 - yf_im[1,2,:]*s1) \
           + yf_re[0,1,:] - yf_re[2,1,:]

    Y_im = yf_im[1,2,:]*c3 + yf_re[1,2,:]*s3 \
           - (yf_im[1,0,:]*c3 + yf_re[1,0,:]*s3) \
           + yf_im[2,1,:]*c2 + yf_re[2,1,:]*s2 \
           - (yf_im[0,1,:]*c2 + yf_re[0,1,:]*s2) \
           + yf_im[1,0,:]*c1 + yf_re[1,0,:]*s1 \
           - (yf_im[1,2,:]*c1 + yf_re[1,2,:]*s1) \
           + yf_im[0,1,:] - yf_im[2,1,:]

    Z_re = yf_re[2,0,:]*c3 - yf_im[2,0,:]*s3 \
           - (yf_re[2,1,:]*c3 - yf_im[2,1,:]*s3) \
           + yf_re[0,2,:]*c2 - yf_im[0,2,:]*s2 \
           - (yf_re[1,2,:]*c2 - yf_im[1,2,:]*s2) \
           + yf_re[2,1,:]*c1 - yf_im[2,1,:]*s1 \
           - (yf_re[2,0,:]*c1 - yf_im[2,0,:]*s1) \
           + yf_re[1,2,:] - yf_re[0,2,:]

    Z_im = yf_im[1,2,:]*c3 + yf_re[1,2,:]*s3 \
           - (yf_im[1,0,:]*c3 + yf_re[1,0,:]*s3) \
           + yf_im[2,1,:]*c2 + yf_re[2,1,:]*s2 \
           - (yf_im[0,1,:]*c2 + yf_re[0,1,:]*s2) \
           + yf_im[1,0,:]*c1 + yf_re[1,0,:]*s1 \
           - (yf_im[1,2,:]*c1 + yf_re[1,2,:]*s1) \
           + yf_im[0,1,:] - yf_im[2,1,:]

    return ((X_re, X_im), (Y_re, Y_im), (Z_re, Z_im))

def AET_XYZ(X_re, X_im, Y_re, Y_im, Z_re, Z_im):
    A_re = 1.0/3.0*(2.0*X_re - Y_re - Z_re)
    A_im = 1.0/3.0*(2.0*X_im - Y_im - Z_im)

    E_re = 1.0/sqrt(3.0)*(Z_re - Y_re)
    E_im = 1.0/sqrt(3.0)*(Z_im - Y_im)

    T_re = 1.0/3.0*(X_re + Y_re + Z_re)
    T_im = 1.0/3.0*(X_im + Y_im + Z_im)

    return ((A_re, A_im), (E_re, E_im), (T_re, T_im))

def make_model(f0_prior, sigma_f0_prior, sigma, hbin, lnAlow, lnAhigh):
    with pm.Model() as model:
        f0_unit = pm.Normal('f0-unit', mu=0, sigma=1)
        f0 = pm.Deterministic('f0', f0_prior + sigma_f0_prior*f0_unit)

        fdot_unit = pm.Normal('fdot-unit', mu=0.0, sigma=1.0)
        fdot = pm.Deterministic('fdot', sigma_f0_prior/Tobs*fdot_unit)

        cos_iota = pm.Uniform('cos_iota', lower=-1, upper=1)
        iota = pm.Deterministic('iota', tt.arccos(cos_iota))

        # TODO: Need to sample efficiently on the torus that is 2*psi + phi and 2*psi - phi
        # This 2-vector gives phi0
        n_phi = pm.Normal('n_phi', mu=zeros(2), sigma=ones(2), shape=(2,))
        phi0 = pm.Deterministic('phi0', tt.arctan2(n_phi[1], n_phi[0]))

        # This 2-vector gives 2*psi
        n_2psi = pm.Normal('n_2psi', mu=zeros(2), sigma=ones(2), shape=(2,))
        psi = pm.Deterministic('psi', tt.arctan2(n_2psi[1], n_2psi[0])/2)

        n_ra_dec = pm.Normal('n_ra_dec', mu=zeros(3), sigma=ones(3), shape=(3,))
        nhat = pm.Deterministic('nhat', n_ra_dec / pmm.sqrt(tt.tensordot(n_ra_dec, n_ra_dec, axes=1)))
        _ = pm.Deterministic('phi', tt.arctan2(n_ra_dec[1], n_ra_dec[0]))
        _ = pm.Deterministic('theta', tt.arccos(nhat[2]))

        lnA = pm.Uniform('lnA', lower=lnAlow, upper=lnAhigh)
        A = pm.Deterministic('A', pmm.exp(lnA))

        y_re, y_im = y_fd(0.0, Tobs, f0, fdot, phi0, nhat, cos_iota, psi, hbin, N)
        ((X_re, X_im), (Y_re, Y_im), (Z_re, Z_im)) = XYZ_freq(y_re, y_im, Tobs, hbin, N)
        ((A_re, A_im), (E_re, E_im), (T_re, T_im)) = AET_XYZ(X_re, X_im, Y_re, Y_im, Z_re, Z_im)

        A_re = pm.Deterministic('A_re', A*A_re)
        A_im = pm.Deterministic('A_im', A*A_im)
        E_re = pm.Deterministic('E_re', A*E_re)
        E_im = pm.Deterministic('E_im', A*E_im)

        snr = pm.Deterministic('SNR', tt.sum(tt.square(A_re/sigma)) + tt.sum(tt.square(A_im/sigma)) + tt.sum(tt.square(E_re/sigma)) + tt.sum(tt.square(E_im/sigma)))

        _ = pm.Normal('A_re_obs', mu=A_re, sigma=sigma, observed=A_re_data, shape=(A_re_data.shape[0],))
        _ = pm.Normal('A_im_obs', mu=A_im, sigma=sigma, observed=A_im_data, shape=(A_im_data.shape[0],))
        _ = pm.Normal('E_re_obs', mu=E_re, sigma=sigma, observed=E_re_data, shape=(E_re_data.shape[0],))
        _ = pm.Normal('E_im_obs', mu=E_im, sigma=sigma, observed=E_im_data, shape=(E_im_data.shape[0],))

    return model

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--seed', metavar='N', type=int, help='specify data generating RNG seed for reproducability')
    parser.add_argument('--sample-seed', metavar='N', type=int, help='specify sampling RNG seed for full reproducability')

    datag = parser.add_argument_group('Data')
    datag.add_argument('--data', metavar='FNAME', help='file containing AET data')
    datag.add_argument('--zeronoise', action='store_true', help='generate no noise')
    parser.add_argument('--Tobs', metavar='T', default=pi*1e7, type=float, help='Tobs (s; default: %(default).2g s)')

    injg = parser.add_argument_group('Injection')
    injg.add_argument('--injfile', metavar='FNAME', help='file containing injection parameters for generated waveform')

    priorg = parser.add_argument_group('Prior')
    priorg.add_argument('--f0', required=True, metavar='F0', type=float, help='guess at frequency')
    priorg.add_argument('--sigma-f0', required=True, metavar='SF0', type=float, help='prior width for frequency')
    priorg.add_argument('--Amin', metavar='AMIN', type=float, default=4.1e-25, help='minimum amplitude in prior (default: %(default).2g)')
    priorg.add_argument('--Amax', metavar='AMAX', type=float, default=4.1e-22, help='maximum amplitude in prior (default: %(default).2g)')

    sampg = parser.add_argument_group('Sampling')
    sampg.add_argument('--draws', metavar='N', default=1000, type=int, help='number of samples to draw (default: %(default)s)')
    sampg.add_argument('--tune', metavar='N', type=int, help='number of tuning samples (default: same as --draws)')
    sampg.add_argument('--chains', metavar='N', type=int, default=3, help='number of independent chains (default: %(default)s)')
    sampg.add_argument('--cores', metavar='N', type=int, default=3, help='number of cores to use for sampling (default: %(default)s)')

    outg = parser.add_argument_group('Output')
    outg.add_argument('--outfile', metavar='FNAME', default='chain.nc', help='output file (default: %(default)s)')

    args = parser.parse_args()

    if args.tune is None:
        tune = args.draws
    else:
        tune = args.tune

    Tobs = args.Tobs
    f0 = args.f0
    sigma_fdot = args.sigma_f0/Tobs

    hbin = int(round(f0*Tobs))
    N = next_pow_two(int(round(4*Tobs*slow_bw(f0, 3*sigma_fdot, Tobs))))

    sigma = sqrt(Tobs*Sn(f0)/4.0)

    if args.data is not None:
        data = genfromtxt(args.data, names=True)
        fs = data['fs']
        A_re_data = data['A_re']
        A_im_data = data['A_im']
        E_re_data = data['E_re']
        E_im_data = data['E_im']
    elif args.zeronoise:
        A_re_data = zeros(N)
        A_im_data = zeros(N)
        E_re_data = zeros(N)
        E_im_data = zeros(N)
    else:
        rstate = np.random.get_state()
        if args.seed is not None:
            np.random.seed(args.seed)
        A_re_data = randn(N)*sigma
        A_im_data = randn(N)*sigma
        E_re_data = randn(N)*sigma
        E_im_data = randn(N)*sigma
        if args.seed is not None:
            np.random.set_state(rstate)

    if args.injfile is not None:
        inj = genfromtxt(args.injfile, names=True)

        th = inj['theta']
        ph = inj['phi']

        nhat = array([cos(ph)*sin(th),
                      sin(ph)*sin(th),
                      cos(th)])
        y_re, y_im = y_fd(0.0, Tobs, inj['f0'], inj['fdot'], inj['phi0'], nhat, inj['cos_iota'], inj['psi'], hbin, N)
        ((X_re, X_im), (Y_re, Y_im), (Z_re, Z_im)) = XYZ_freq(y_re, y_im, Tobs, hbin, N)
        ((A_re, A_im), (E_re, E_im), _) = AET_XYZ(X_re, X_im, Y_re, Y_im, Z_re, Z_im)

        A_re *= inj['A']
        A_im *= inj['A']
        E_re *= inj['A']
        E_im *= inj['A']

        A_re = A_re.eval()
        A_im = A_im.eval()
        E_re = E_re.eval()
        E_im = E_im.eval()

        A_re_data += A_re
        A_im_data += A_im
        E_re_data += E_re
        E_im_data += E_im

    model = make_model(f0, args.sigma_f0, sigma, hbin, log(args.Amin), log(args.Amax))

    rstate = np.random.get_state()
    if args.sample_seed is not None:
        np.random.seed(args.sample_seed)
    with model:
        trace = pm.sample(draws=args.draws, tune=tune, chains=args.chains, cores=args.cores)
    if args.sample_seed is not None:
        np.random.set_state(rstate)

    fit = az.from_pymc3(trace)

    ofile = args.outfile + '.tmp'
    if op.exists(ofile):
        os.remove(ofile)
    az.to_netcdf(fit, ofile)
    os.rename(ofile, args.outfile)
