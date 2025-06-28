# PyROOT-Emulator

A comprehensive Python implementation of ROOT's core functionality using NumPy and Matplotlib, designed to provide ROOT-like analysis capabilities without requiring the full ROOT installation.

## 🚀 Features

### Core ROOT Classes
- **TFile**: ROOT file I/O with real ROOT file support via uproot
- **TTree**: Tree data structure with branch access and selection parsing
- **TH1F**: 1D histograms with full statistical functions
- **TF1**: 1D function objects with fitting capabilities
- **TCanvas**: Canvas for plotting with interactive display
- **TGraph/TGraphErrors**: Graph objects with error bars
- **TLegend**: Plot legends with ROOT-style positioning
- **TLatex**: LaTeX text rendering for plots

### Mathematical Functions (TMath)
- **Statistical Functions**: Gaussian, Landau, Poisson distributions
- **Mathematical Operations**: Trigonometric, logarithmic, exponential
- **Special Functions**: Gamma, Beta, Factorial, Binomial coefficients
- **Utility Functions**: Min/Max, sorting, range constraints

### Advanced Features
- **Real ROOT File Support**: Seamless integration with uproot for reading .root files
- **Selection String Parsing**: ROOT-style cut expressions (`"pt > 20 && abs(eta) < 2.5"`)
- **Complex Expression Evaluation**: Support for mathematical expressions in TTree::Draw
- **Histogram Fitting**: Scipy-based curve fitting with chi-square statistics
- **Color Mapping**: Complete ROOT color palette mapping to matplotlib
- **Interactive Plotting**: Popup windows with Qt5Agg/TkAgg backends
- **Histogram Serialization**: Save/load histograms to/from JSON

## 📦 Installation

### Dependencies
```bash
pip install numpy matplotlib scipy pandas
```

### Optional Dependencies (for enhanced functionality)
```bash
# For real ROOT file support
pip install uproot awkward

# For interactive plotting
pip install PyQt5  # or tkinter (usually included with Python)
```

### Usage
Simply download `ROOT.py` and import it in your Python scripts:

```python
from ROOT import *
```

## 🔥 Quick Start

```python
from ROOT import *
import numpy as np

# Create canvas and histogram
canvas = TCanvas("c1", "Demo", 800, 600)
hist = TH1F("h1", "Gaussian Distribution;X;Counts", 100, -5, 5)

# Fill with random data
for i in range(10000):
    hist.Fill(gRandom.Gaus(0, 1))

# Style and draw
hist.SetLineColor(kBlue)
hist.SetFillColor(kAzure - 9)
hist.Draw()

# Fit with Gaussian
gauss = TF1("gauss", "gaus", -5, 5)
hist.Fit(gauss)
gauss.Draw("SAME")

# Save plot
canvas.SaveAs("demo.png")
```

## 📊 Comparison with Official PyROOT

| Feature | PyROOT Alternative | Official PyROOT | Notes |
|---------|-------------------|-----------------|-------|
| **Installation** | ✅ pip install only | ❌ Complex ROOT build | No ROOT compilation needed |
| **File Size** | ✅ Single 50KB file | ❌ Multi-GB installation | Lightweight solution |
| **Dependencies** | ✅ Pure Python stack | ❌ C++ ROOT + Python bindings | Standard scientific Python |
| **Performance** | ⚠️ Python speed | ✅ C++ performance | Good for analysis, not HPC |
| **ROOT File Reading** | ✅ Via uproot | ✅ Native | Both support .root files |
| **Plotting Backend** | ✅ Matplotlib | ✅ ROOT's graphics | More familiar to Python users |
| **Interactive Analysis** | ✅ Jupyter friendly | ✅ ROOT prompt | Better notebook integration |
| **Learning Curve** | ✅ Python-centric | ⚠️ ROOT-specific | Easier for Python developers |
| **Feature Coverage** | ⚠️ Core features | ✅ Complete ROOT | Covers 80% of common use cases |

## 🎯 Core Functionality

### Histograms
```python
# Create and fill histogram
hist = TH1F("data", "My Data;X;Counts", 50, 0, 100)
for value in data:
    hist.Fill(value)

# Statistics
print(f"Mean: {hist.GetMean():.2f}")
print(f"RMS: {hist.GetRMS():.2f}")
print(f"Entries: {hist.GetEntries()}")

# Styling
hist.SetLineColor(kRed)
hist.SetFillColorAlpha(kBlue, 0.3)
hist.Draw()
```

### Function Fitting
```python
# Create function and fit
func = TF1("fit", "gaus", -5, 5)
hist.Fit(func)

# Access fit results
print(f"Chi2/NDF: {func.GetChisquare():.2f}/{func.GetNDF()}")
print(f"Amplitude: {func.GetParameter(0):.2f}")
print(f"Mean: {func.GetParameter(1):.4f}")
print(f"Sigma: {func.GetParameter(2):.4f}")
```

### ROOT File Analysis
```python
# Open ROOT file (requires uproot)
file = TFile.Open("data.root")
tree = file.Get("events")

# Create histogram from tree
hist = TH1F("mass", "Invariant Mass;Mass [GeV];Events", 100, 2.8, 3.2)
tree.Draw("mass >> hist", "pt > 20 && abs(eta) < 2.5")

# Fit resonance
gauss = TF1("signal", "gaus", 2.9, 3.1)
hist.Fit(gauss, "R")
```

### Advanced Selection Parsing
```python
# Complex selection strings
selection = "pt > 20 && abs(eta) < 2.5 && mass > 2.8 && mass < 3.2"
tree.Draw("pt >> pt_hist", selection)

# Mathematical expressions
tree.Draw("sqrt(px*px + py*py) >> pt_calc", "energy > 50")
```

### Graphs and Error Bars
```python
# Create graph with errors
graph = TGraphErrors()
for i, (x, y, ex, ey) in enumerate(data_points):
    graph.SetPoint(i, x, y)
    graph.SetPointError(i, ex, ey)

graph.SetMarkerStyle(kFullCircle)
graph.SetMarkerColor(kBlue)
graph.Draw("AP")
```

## 🎨 Styling and Colors

Complete ROOT color palette support:
```python
# ROOT colors
hist.SetLineColor(kRed + 1)
hist.SetFillColor(kAzure - 9)
hist.SetMarkerColor(kBlue)

# Line styles
hist.SetLineStyle(kDashed)

# Marker styles
graph.SetMarkerStyle(kFullCircle)
```

## 📈 Mathematical Functions

Full TMath library implementation:
```python
# Statistical functions
prob = TMath.Gaus(x, mean, sigma)
landau = TMath.Landau(x, mpv, sigma)

# Mathematical operations
result = TMath.Sqrt(TMath.Power(x, 2) + TMath.Power(y, 2))
angle = TMath.ATan2(y, x)

# Special functions
gamma_val = TMath.Gamma(z)
factorial = TMath.Factorial(n)
```

## 💾 Data Persistence

```python
# Save histogram to JSON
hist.Save("data.json")

# Load histogram from JSON
hist2 = TH1F.Load("data.json")

# ROOT file I/O (with uproot)
file = TFile.Open("analysis.root")
tree = file.Get("events")
```

## 🔧 Configuration

### Interactive Backend Setup
```python
# Automatic backend selection for interactive plots
setup_interactive_matplotlib()  # Called automatically

# Manual backend selection
import matplotlib
matplotlib.use('Qt5Agg')  # or 'TkAgg'
```

### Customization
```python
# Custom color schemes
ROOTConstants.colors[100] = 'purple'

# Custom function definitions
custom_func = TF1("custom", "[0]*x*x + [1]*x + [2]", 0, 10)
```

## 🎯 Use Cases

### Perfect For:
- **Physics Analysis**: Histogram fitting, peak finding, statistical analysis
- **Teaching**: ROOT concepts without installation complexity
- **Prototyping**: Quick analysis scripts and Jupyter notebooks
- **Cross-platform**: Works wherever Python runs
- **Data Visualization**: Publication-quality plots with matplotlib

### Not Ideal For:
- **High-performance Computing**: Large-scale data processing
- **Advanced ROOT Features**: Complex geometry, Monte Carlo simulation
- **Legacy Code**: Direct porting of complex ROOT macros
- **Real-time Analysis**: Performance-critical applications

## 📚 Documentation

### Class Methods
Each class implements the most commonly used ROOT methods:

- **TH1F**: `Fill()`, `Draw()`, `Fit()`, `GetMean()`, `GetRMS()`, `Integral()`
- **TF1**: `SetParameters()`, `Draw()`, `GetParameter()`, `GetChisquare()`
- **TTree**: `Draw()`, `Get()`, expression evaluation
- **TCanvas**: `SaveAs()`, `Show()`, multiple plot management

### Error Handling
- Graceful fallbacks for missing dependencies
- Informative error messages for debugging
- Automatic backend selection for plotting

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional ROOT classes (TH2F, TProfile, etc.)
- More mathematical functions
- Enhanced ROOT file compatibility
- Performance optimizations
- Additional plotting options

## 📄 License

MIT License - feel free to use in your projects!

## 🙏 Acknowledgments

- **ROOT Team**: For the original ROOT framework
- **Uproot Team**: For excellent ROOT file I/O
- **Scientific Python Community**: NumPy, Matplotlib, SciPy
- **Physics Community**: For feedback and use cases

---

*This implementation aims to provide 80% of ROOT's functionality with 20% of the complexity, making it perfect for most physics analysis tasks in a pure Python environment.*
