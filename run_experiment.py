"""Run the full experiment:
  1. load clean sources from ./sources/*.wav (or synthesize stand-ins),
  2. mix with your own mixing matrix,
  3. center + PCA-whiten + FastICA (from scratch),
  4. save separated audio, plots, metrics (PCA-only vs ICA, SNR sweep, nonlinearity sweep).
"""
import glob, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly
import ica as myica

SR = 16000
SECONDS = 10
OUT = "outputs"
os.makedirs(f"{OUT}/mixed", exist_ok=True)
os.makedirs(f"{OUT}/separated_ica", exist_ok=True)
os.makedirs(f"{OUT}/separated_pca", exist_ok=True)
os.makedirs(f"{OUT}/originals", exist_ok=True)
rng = np.random.default_rng(42)


def load_wav(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(np.asarray(x).dtype, np.integer) or np.abs(x).max() > 1:
        x = x / (np.abs(x).max() + 1e-12)
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    x = x[: SR * SECONDS]
    return x / (np.abs(x).max() + 1e-12)


def synth_sources(T):
    """Stand-ins used only if ./sources is empty (so the pipeline always runs)."""
    t = np.arange(T) / SR
    # "music": melody of harmonic tones
    notes = np.array([261.6, 329.6, 392.0, 440.0, 349.2, 293.7])
    f = notes[(t * 3).astype(int) % len(notes)]
    phase = 2 * np.pi * np.cumsum(f) / SR
    music = sum(np.sin(h * phase) / h for h in (1, 2, 3)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    # "speech-like": formant-ish amplitude-modulated sawtooth with pauses
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.7 * t)
    saw = 2 * ((np.cumsum(f0) / SR) % 1) - 1
    env = (np.sin(2 * np.pi * 2.2 * t) > -0.2) * (0.5 + 0.5 * np.sin(2 * np.pi * 5 * t) ** 2)
    speech = saw * env
    # "ambient": coloured heavy-tailed noise (rain-like)
    n = rng.laplace(size=T)
    ambient = np.convolve(n, np.ones(8) / 8, mode="same")
    S = np.vstack([music, speech, ambient])
    return S / np.abs(S).max(axis=1, keepdims=True)


# ---------- 1. sources ----------
files = sorted(glob.glob("sources/*.wav"))
if files:
    srcs = [load_wav(f) for f in files]
    T = min(len(s) for s in srcs)
    S_true = np.vstack([s[:T] for s in srcs])
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
else:
    print("No wavs in ./sources, using synthetic stand-ins.")
    S_true = synth_sources(SR * SECONDS)
    names = ["music_synth", "speech_synth", "ambient_synth"]
N, T = S_true.shape
print(f"{N} sources, {T} samples ({T/SR:.1f}s @ {SR} Hz): {names}")
for n_, s in zip(names, S_true):
    wavfile.write(f"{OUT}/originals/{n_}.wav", SR, (s * 32767).astype(np.int16))

# ---------- 2. mixing ----------
A = rng.uniform(0.3, 1.0, size=(N, N))          # own mixing matrix (M = N mixtures)
while np.linalg.cond(A) > 20:
    A = rng.uniform(0.3, 1.0, size=(N, N))
X = A @ S_true
np.savetxt(f"{OUT}/mixing_matrix.txt", A, fmt="%.4f")
for i, x in enumerate(X):
    wavfile.write(f"{OUT}/mixed/mix_{i+1}.wav", SR, (x / np.abs(x).max() * 32767 * 0.95).astype(np.int16))


def save_set(folder, S, prefix):
    for i, s in enumerate(S):
        wavfile.write(f"{OUT}/{folder}/{prefix}_{i+1}.wav", SR, (s / np.abs(s).max() * 32767 * 0.95).astype(np.int16))


# ---------- 3. ICA vs PCA ----------
S_ica, W_tot, n_iter = myica.ica(X, nonlin="logcosh", seed=0)
Xc, _ = myica.center(X)
Z, Wwh, E, d = myica.pca_whiten(Xc)             # PCA components = whitened, only decorrelated
S_pca = Z

res_ica = myica.evaluate(S_true, S_ica)
res_pca = myica.evaluate(S_true, S_pca)
save_set("separated_ica", res_ica["est"], "ica_est")
save_set("separated_pca", res_pca["est"], "pca_est")
print(f"ICA converged in {n_iter} iterations")
print("ICA  corr", np.round(res_ica["corr"], 3), "SDR", np.round(res_ica["sdr"], 1), "dB")
print("PCA  corr", np.round(res_pca["corr"], 3), "SDR", np.round(res_pca["sdr"], 1), "dB")

# sanity: whitened covariance ~ identity
print("Cov of whitened data:\n", np.round(np.cov(Z), 3))

results = {"sources": names, "n_iter": int(n_iter),
           "ica": {k: res_ica[k] for k in ("corr", "sdr", "mean_corr", "mean_sdr")},
           "pca": {k: res_pca[k] for k in ("corr", "sdr", "mean_corr", "mean_sdr")}}

# ---------- 4. experiments ----------
# (a) SNR sweep: add Gaussian sensor noise to mixtures
snrs = [40, 30, 20, 10, 5, 0]
sweep = {"snr": snrs, "ica": [], "pca": []}
for snr in snrs:
    ica_s, pca_s = [], []
    for trial in range(5):
        r = np.random.default_rng(100 + trial)
        P = np.mean(X ** 2, axis=1, keepdims=True)
        Xn = X + r.standard_normal(X.shape) * np.sqrt(P / 10 ** (snr / 10))
        Si, _, _ = myica.ica(Xn, seed=trial)
        Zn = myica.pca_whiten(myica.center(Xn)[0])[0]
        ica_s.append(myica.evaluate(S_true, Si)["mean_sdr"])
        pca_s.append(myica.evaluate(S_true, Zn)["mean_sdr"])
    sweep["ica"].append(float(np.mean(ica_s))); sweep["pca"].append(float(np.mean(pca_s)))
results["snr_sweep"] = sweep

# (b) non-linearity comparison
nl = {}
for name in ("logcosh", "exp", "cube"):
    s, _, it = myica.ica(X, nonlin=name, seed=0)
    nl[name] = {"mean_sdr": myica.evaluate(S_true, s)["mean_sdr"], "iters": int(it)}
results["nonlinearity"] = nl
print("Nonlinearity comparison:", nl)

# (c) non-Gaussianity: kurtosis of sources / PCA comps / ICA comps
def kurt(x):
    x = (x - x.mean()) / x.std()
    return float(np.mean(x ** 4) - 3)
results["excess_kurtosis"] = {
    "sources": [kurt(s) for s in S_true], "mixtures": [kurt(x) for x in X],
    "pca": [kurt(s) for s in res_pca["est"]], "ica": [kurt(s) for s in res_ica["est"]]}
print("Excess kurtosis:", {k: np.round(v, 2).tolist() for k, v in results["excess_kurtosis"].items()})

json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)

# ---------- 5. plots ----------
t = np.arange(T) / SR
sl = slice(0, SR * 2)
fig, ax = plt.subplots(N, 4, figsize=(16, 2.2 * N), sharex=True)
cols = [("Original sources", S_true), ("Mixtures", X), ("PCA output", res_pca["est"]), ("ICA output", res_ica["est"])]
for j, (title, M) in enumerate(cols):
    for i in range(N):
        ax[i, j].plot(t[sl], M[i, sl], lw=0.5)
        ax[i, j].set_yticks([])
        if i == 0: ax[i, j].set_title(title)
ax[-1, 0].set_xlabel("time (s)")
plt.tight_layout(); plt.savefig(f"{OUT}/waveforms.png", dpi=110); plt.close()

plt.figure(figsize=(6, 4))
plt.plot(snrs, sweep["ica"], "o-", label="ICA (FastICA, from scratch)")
plt.plot(snrs, sweep["pca"], "s--", label="PCA only")
plt.xlabel("Mixture SNR (dB)"); plt.ylabel("Mean SDR (dB)"); plt.gca().invert_xaxis()
plt.grid(alpha=.3); plt.legend(); plt.title("Separation quality vs noise")
plt.tight_layout(); plt.savefig(f"{OUT}/snr_sweep.png", dpi=110); plt.close()

# scatter: joint distribution (2 sources) -> shows why PCA can't unmix
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for a_, (title, M) in zip(ax, [("Sources", S_true), ("Mixtures", X), ("After PCA whitening", Z), ("After ICA", S_ica)]):
    a_.scatter(M[0, ::20], M[1, ::20], s=1, alpha=.4); a_.set_title(title); a_.set_aspect("equal", "datalim")
plt.tight_layout(); plt.savefig(f"{OUT}/scatter.png", dpi=110); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
for a_, (title, M) in zip(ax, [("Sources", S_true), ("Mixtures", X), ("ICA output", res_ica["est"])]):
    a_.hist(M[0], bins=80, density=True); a_.set_title(f"{title} - histogram of signal 1")
plt.tight_layout(); plt.savefig(f"{OUT}/histograms.png", dpi=110); plt.close()
print("Done. See outputs/")
