# PyROOT-Emulator

This project is for students, researchers, or anyone learning particle/data physics who:
- Wants to practice ROOT-style analysis without installing the full C++ ROOT framework.
- Works on low-spec systems (e.g., academic laptops) that can't handle ROOT.
- Needs histogramming, fitting, and plotting capabilities using just Python + matplotlib.
- Is learning PyROOT syntax but wants a “sandbox” to understand the logic behind `TH1F`, `TF1`, `TCanvas`, etc.

---

## ⚙️ What It Does

This project replicates a minimal subset of ROOT's core histogram and fitting workflow — enough for educational or demo purposes.

### ✅ Features

- `TH1F`-style histograms:
  - Custom binning, titles, axis labels
  - Data filling from arrays or synthetic distributions
  - `FillRandom()` method to generate histograms from mathematical functions
  - Optional error bars and styling (line, fill color, marker size)

- `TF1`-style function class:
  - Supports ROOT-style formulas like `[0]*exp(-0.5*((x-[1])/[2])**2)`
  - Set/get parameters via `SetParameters()`, `GetParameter()`
  - Draw plots with or without histograms using `"SAME"` option
  - `GetMaximumEstimate()` used for rejection sampling

- Fitting support:
  - Use `.Fit()` on a histogram to fit it with a `TF1` function
  - Computes fit parameters, errors, χ²/ndf
  - Uses `scipy.optimize.curve_fit` under the hood
  - Visualizes both data and fitted curve with matplotlib

- Output:
  - Automatically saves plots as `.png` using a `TCanvas` wrapper
  - Replaces interactive ROOT GUI with static image output

---

## 📌 Example Usage

```python
import ROOT

# Read data from file
data = [float(line.strip()) for line in open("data.txt") if line.strip()]

# Create histogram and fill with data
hist = ROOT.TH1F("hist", "Random Data Histogram;Value;Frequency", 50, 0, 10)
for value in data:
    hist.Fill(value)

# Fit with a Gaussian
fit_func = ROOT.TF1.builtin("gaus")
hist.Fit(fit_func)

# Draw histogram and fit
hist.Draw("HIST")
fit_func.Draw("SAME")

# Save to file
canvas = ROOT.TCanvas("c", "Histogram Canvas", 800, 600)
canvas.SaveAs("histogram.png")
```

## 🔐 Security Note on `eval()`

This project uses Python’s `eval()` to interpret formulas in `TF1`, for example:

```python
"[0]*exp(-0.5*((x-[1])/[2])**2)"
```

Internally, expressions like this are evaluated as:

```python
eval(formula, {"np": numpy, "x_val": x_val, "__builtins__": {}})
```

Although the `__builtins__` dictionary is disabled to reduce risk, **this is not fully secure**. If you're exposing this code to untrusted users (e.g., web apps, input forms), `eval()` could still be exploited with clever input.

### ⚠️ Never use `eval()` with:

* User-defined input directly (from `input()`, web forms, etc.)
* Any system where malicious code could be injected

### ✅ Safe if:

* You're using it offline for personal or academic purposes
* You **do not** let others pass custom formulas into `TF1`
