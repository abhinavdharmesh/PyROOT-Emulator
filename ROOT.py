import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.optimize import curve_fit
import pandas as pd
import os
import re
import json
import warnings
from typing import Optional, List, Dict, Any, Union
from collections import defaultdict
import ast
import operator

# Try to import uproot for real ROOT file support
try:
    import uproot
    import awkward as ak
    UPROOT_AVAILABLE = True
    print("[INFO] Uproot available - real ROOT file support enabled")
except ImportError:
    UPROOT_AVAILABLE = False
    print("[INFO] Uproot not available - using simulation mode")

class gRandom:
    @staticmethod
    def Gaus(mean=0, sigma=1):
        return np.random.normal(mean, sigma)
    
    @staticmethod
    def Uniform(xmin=0, xmax=1):
        return np.random.uniform(xmin, xmax)
    
    @staticmethod
    def Poisson(mean):
        return np.random.poisson(mean)

class ROOTColors:
    """ROOT color scheme mapping"""
    colors = {
        1: 'black', 2: 'red', 3: 'green', 4: 'blue', 5: 'yellow',
        6: 'magenta', 7: 'cyan', 8: 'darkgreen', 9: 'purple', 
        10: 'white', 11: 'gray', 12: 'brown', 13: 'pink',
        14: 'gold', 15: 'lightblue', 16: 'lightgreen'
    }
    
    @classmethod
    def get_color(cls, color_index):
        return cls.colors.get(color_index, 'black')
        
class ROOTConstants:
    """ROOT color and style constants"""
    
    # Color constants
    kWhite = 0
    kBlack = 1
    kGray = 920
    kRed = 632
    kGreen = 416
    kBlue = 600
    kYellow = 400
    kMagenta = 616
    kCyan = 432
    kOrange = 800
    kSpring = 820
    kTeal = 840
    kAzure = 860
    kViolet = 880
    kPink = 900
    
    # Line style constants
    kSolid = 1
    kDashed = 2
    kDotted = 3
    kDashDotted = 4
    
    # Marker style constants
    kDot = 1
    kPlus = 2
    kStar = 3
    kCircle = 4
    kMultiply = 5
    kFullDotSmall = 6
    kFullDotMedium = 7
    kFullDotLarge = 8
    kFullCircle = 20
    kFullSquare = 21
    kFullTriangleUp = 22
    kFullTriangleDown = 23
    kOpenCircle = 24
    kOpenSquare = 25
    kOpenTriangleUp = 26
    kOpenDiamond = 27
    kOpenCross = 28
    kFullStar = 29
    kOpenStar = 30
    
    # Fill style constants
    kFEmpty = 0
    kFSolid = 1001
    kFHollow = 0
    
    @classmethod
    def get_mpl_color(cls, root_color):
        """Convert ROOT color constants to matplotlib colors"""
        color_map = {
            cls.kWhite: 'white', cls.kBlack: 'black', cls.kGray: 'gray',
            cls.kRed: 'red', cls.kGreen: 'green', cls.kBlue: 'blue',
            cls.kYellow: 'yellow', cls.kMagenta: 'magenta', cls.kCyan: 'cyan',
            cls.kOrange: 'orange', cls.kSpring: 'springgreen', cls.kTeal: 'teal',
            cls.kAzure: 'azure', cls.kViolet: 'violet', cls.kPink: 'pink',
            # Handle +/- variations
            cls.kBlue + 1: 'darkblue', cls.kBlue - 1: 'lightblue',
            cls.kRed + 1: 'darkred', cls.kRed - 1: 'lightcoral',
            cls.kGreen + 1: 'darkgreen', cls.kGreen - 1: 'lightgreen',
            cls.kAzure - 9: 'lightcyan', cls.kAzure + 1: 'darkturquoise',
            cls.kOrange + 1: 'darkorange', cls.kOrange - 1: 'moccasin',
        }
        return color_map.get(root_color, 'black')
    
    @classmethod
    def get_mpl_linestyle(cls, root_style):
        """Convert ROOT line styles to matplotlib"""
        style_map = {
            cls.kSolid: '-', cls.kDashed: '--', 
            cls.kDotted: ':', cls.kDashDotted: '-.'
        }
        return style_map.get(root_style, '-')
    
    @classmethod  
    def get_mpl_marker(cls, root_marker):
        """Convert ROOT markers to matplotlib"""
        marker_map = {
            cls.kDot: '.', cls.kPlus: '+', cls.kStar: '*',
            cls.kCircle: 'o', cls.kMultiply: 'x', cls.kFullCircle: 'o',
            cls.kFullSquare: 's', cls.kFullTriangleUp: '^', 
            cls.kFullTriangleDown: 'v', cls.kOpenCircle: 'o',
            cls.kOpenSquare: 's', cls.kOpenTriangleUp: '^',
            cls.kOpenDiamond: 'D', cls.kOpenCross: '+',
            cls.kFullStar: '*', cls.kOpenStar: '*'
        }
        return marker_map.get(root_marker, 'o')
class SelectionParser:
    """Enhanced selection string parser for ROOT-style cuts"""
    
    # Safe operations for eval
    SAFE_OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
        ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne,
        ast.And: operator.and_, ast.Or: operator.or_, ast.Not: operator.not_,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
    }
    
    @classmethod
    def parse_selection(cls, selection_str, data_dict):
        """Safely parse and evaluate ROOT-style selection strings"""
        if not selection_str.strip():
            return np.ones(len(list(data_dict.values())[0]), dtype=bool)
        
        # Replace ROOT operators with Python equivalents
        selection_str = selection_str.replace('&&', ' and ')
        selection_str = selection_str.replace('||', ' or ')
        selection_str = selection_str.replace('!', ' not ')
        
        try:
            # Parse as AST for safety
            tree = ast.parse(selection_str, mode='eval')
            return cls._eval_node(tree.body, data_dict)
        except Exception as e:
            print(f"[WARNING] Selection parsing failed: {e}")
            return np.ones(len(list(data_dict.values())[0]), dtype=bool)
    
    @classmethod
    def _eval_node(cls, node, data_dict):
        """Recursively evaluate AST nodes"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in data_dict:
                return data_dict[node.id]
            else:
                raise ValueError(f"Unknown variable: {node.id}")
        elif isinstance(node, ast.BinOp):
            left = cls._eval_node(node.left, data_dict)
            right = cls._eval_node(node.right, data_dict)
            op = cls.SAFE_OPS[type(node.op)]
            return op(left, right)
        elif isinstance(node, ast.Compare):
            left = cls._eval_node(node.left, data_dict)
            result = np.ones_like(left, dtype=bool)
            for op, comparator in zip(node.ops, node.comparators):
                right = cls._eval_node(comparator, data_dict)
                op_func = cls.SAFE_OPS[type(op)]
                result = result & op_func(left, right)
                left = right
            return result
        elif isinstance(node, ast.BoolOp):
            values = [cls._eval_node(val, data_dict) for val in node.values]
            op = cls.SAFE_OPS[type(node.op)]
            result = values[0]
            for val in values[1:]:
                result = op(result, val)
            return result
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

class TFile:
    def __init__(self, filename, mode="READ"):
        self.filename = filename
        self.mode = mode
        self.trees = {}
        self.histograms = {}
        self.is_open = False
        self._uproot_file = None
        
    @classmethod
    def Open(cls, filename, mode="READ"):
        file_obj = cls(filename, mode)
        file_obj.is_open = True
        
        # Try to open with uproot first
        if UPROOT_AVAILABLE and os.path.exists(filename):
            try:
                file_obj._uproot_file = uproot.open(filename)
                print(f"[INFO] Opened ROOT file with uproot: {filename}")
                return file_obj
            except Exception as e:
                print(f"[WARNING] Failed to open with uproot: {e}")
        
        print(f"[INFO] Using simulated ROOT file: {filename}")
        return file_obj
    
    def Get(self, name):
        if self._uproot_file and name in self._uproot_file:
            # Real ROOT file
            obj = self._uproot_file[name]
            if hasattr(obj, 'arrays'):  # It's a tree
                return TTree(name, f"Tree {name}", uproot_tree=obj)
            else:
                # Handle histograms, etc.
                return obj
        else:
            # Simulated
            if name not in self.trees:
                self.trees[name] = TTree(name, f"Simulated tree {name}")
            return self.trees[name]
    
    def ls(self):
        """List contents"""
        if self._uproot_file:
            return list(self._uproot_file.keys())
        else:
            return list(self.trees.keys())
    
    def Close(self):
        if self._uproot_file:
            self._uproot_file.close()
        self.is_open = False
        print(f"[INFO] Closed ROOT file: {self.filename}")

class TTree:
    def __init__(self, name, title, uproot_tree=None):
        self.name = name
        self.title = title
        self._uproot_tree = uproot_tree
        self.data = {}
        self._load_data()
        
    def _load_data(self):
        """Load data from uproot tree or generate sample data"""
        if self._uproot_tree:
            try:
                # Load all branches as arrays
                arrays = self._uproot_tree.arrays(library="np")
                self.data = {key: arrays[key] for key in arrays.keys()}
                print(f"[INFO] Loaded {len(self.data)} branches from real ROOT tree")
            except Exception as e:
                print(f"[WARNING] Failed to load uproot data: {e}")
                self._generate_sample_data()
        else:
            self._generate_sample_data()
            
    def _generate_sample_data(self):
        """Generate sample data for demonstration"""
        n_events = 50000
        # Simulate realistic HEP data
        # Signal: resonance at 3.0 GeV with width 0.02 GeV
        signal_frac = 0.15
        n_signal = int(signal_frac * n_events)
        n_background = n_events - n_signal
        
        # Signal events
        mass_signal = np.random.normal(3.0, 0.02, n_signal)
        pt_signal = np.random.exponential(3.0, n_signal) + 1.0
        
        # Background events  
        mass_background = np.random.uniform(2.8, 3.2, n_background)
        pt_background = np.random.exponential(2.0, n_background) + 0.5
        
        # Combine
        mass = np.concatenate([mass_signal, mass_background])
        pt = np.concatenate([pt_signal, pt_background])
        
        # Shuffle
        indices = np.random.permutation(n_events)
        mass = mass[indices]
        pt = pt[indices]
        
        # Add more realistic variables
        eta = np.random.normal(0, 2.0, n_events)
        phi = np.random.uniform(-np.pi, np.pi, n_events)
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
        
        self.data = {
            'mass': mass, 'pt': pt, 'eta': eta, 'phi': phi,
            'px': px, 'py': py, 'pz': pz, 'energy': energy,
            'charge': np.random.choice([-1, 1], n_events),
            'pid': np.random.choice([11, 13, 211, 321], n_events)  # electrons, muons, pions, kaons
        }
        
    def Draw(self, expression, selection="", option=""):
        """Enhanced Draw method with selection parsing"""
        # Handle >>+ syntax for appending
        append_mode = ">>+" in expression
        if append_mode:
            expression = expression.replace(">>+", ">>")
        
        # Parse expression
        if ">>" in expression:
            var_expr, hist_name = expression.split(">>", 1)
            var_expr = var_expr.strip()
            hist_name = hist_name.strip()
            
            # Parse variable expression (could be complex like "sqrt(px*px + py*py)")
            try:
                data = self._evaluate_expression(var_expr)
            except Exception as e:
                print(f"[ERROR] Failed to evaluate expression '{var_expr}': {e}")
                return
            
            # Apply selection
            if selection:
                try:
                    mask = SelectionParser.parse_selection(selection, self.data)
                    data = data[mask]
                    print(f"[INFO] Selection '{selection}' passed {np.sum(mask)}/{len(mask)} events")
                except Exception as e:
                    print(f"[WARNING] Selection failed: {e}")
            
            # Find histogram and fill
            hist = self._find_histogram(hist_name)
            if hist:
                if not append_mode:
                    hist.Reset()  # Clear existing data
                for value in data:
                    if np.isfinite(value):
                        hist.Fill(value)
                print(f"[INFO] Filled histogram {hist_name} with {len(data)} entries")
        else:
            print("[WARNING] Only '>> histogram' syntax supported currently")
    
    def _evaluate_expression(self, expr):
        """Evaluate complex expressions like 'sqrt(px*px + py*py)'"""
        # Replace common ROOT functions
        expr = expr.replace('sqrt', 'np.sqrt')
        expr = expr.replace('abs', 'np.abs')
        expr = expr.replace('sin', 'np.sin')
        expr = expr.replace('cos', 'np.cos')
        expr = expr.replace('exp', 'np.exp')
        expr = expr.replace('log', 'np.log')
        
        # Create safe namespace
        namespace = {'np': np, **self.data}
        
        try:
            return eval(expr, {"__builtins__": {}, "np": np}, namespace)
        except Exception as e:
            # Fallback: try simple variable lookup
            if expr in self.data:
                return self.data[expr]
            else:
                raise e
    
    def _find_histogram(self, hist_name):
        """Find histogram in calling scope"""
        import inspect
        frame = inspect.currentframe()
        try:
            # Look through call stack for the histogram
            for i in range(10):  # Check up to 10 frames up
                frame = frame.f_back
                if frame is None:
                    break
                    
                # Check locals and globals
                for scope in [frame.f_locals, frame.f_globals]:
                    if hist_name in scope and isinstance(scope[hist_name], TH1F):
                        return scope[hist_name]
            return None
        finally:
            del frame
#----------------------------------------------------------------------------------------------#
#import numpy as np
#import re

class TF1:
    _registry = {}

    def __init__(self, name, formula, xmin, xmax):
        self.name = name
        self.formula_raw = formula
        self.formula = self._translate_formula(formula)
        self.xmin = xmin
        self.xmax = xmax
        self.n_params = self._count_parameters(formula)
        self.parameters = [1.0] * self.n_params
        self.parameter_errors = [0.0] * self.n_params
        self.parameter_names = [f"p{i}" for i in range(self.n_params)]
        self.chi2 = 0
        self.ndf = 0
        self._compiled = None

    @classmethod
    def get_by_name(cls, name):
        return cls._registry.get(name)

    def register(self):
        TF1._registry[self.name] = self

    @classmethod
    def builtin(cls, name):
        if name == "gaus":
            f = cls("gaus", "[0]*exp(-0.5*((x-[1])/[2])**2)", -5, 5)
            f.SetParameters(1.0, 0.0, 1.0)
        elif name == "expo":
            f = cls("expo", "[0]*exp([1]*x)", 0, 10)
            f.SetParameters(1.0, -1.0)
        else:
            raise ValueError(f"[TF1 builtin] Unknown function: {name}")
        f.register()
        return f

    def _translate_formula(self, formula):
        # Replace parameters
        for i in range(10):
            formula = formula.replace(f'[{i}]', f'p[{i}]')
        # Replace math functions
        replacements = {
            'exp': 'np.exp',
            'sqrt': 'np.sqrt',
            'pow': 'np.power',
            'sin': 'np.sin',
            'cos': 'np.cos',
            'tan': 'np.tan',
            'log': 'np.log',
            'abs': 'np.abs',
            'pi': 'np.pi',
            'TMath::Pi()': 'np.pi'
        }
        for old, new in replacements.items():
            formula = formula.replace(old, new)
        return formula

    def _count_parameters(self, formula):
        matches = re.findall(r'\[(\d+)\]', formula)
        return max(map(int, matches)) + 1 if matches else 0

    def SetParameters(self, *params):
        self.parameters = list(params)
        self._compile()

    def GetParameter(self, index):
        if 0 <= index < len(self.parameters):
            return self.parameters[index]
        return 0

    def _compile(self):
        """Compile the formula to a fast lambda"""
        code = f"lambda x, p: {self.formula}"
        try:
            self._compiled = eval(code, {"np": np})
        except Exception as e:
            print(f"[TF1 compile error] {e}")
            self._compiled = None

    def _eval_function(self, x, *params):
        if not self._compiled:
            self._compile()
        try:
            return self._compiled(x, params)
        except Exception as e:
            print(f"[TF1 eval error] {e}")
            return np.zeros_like(x)

    def GetMaximumEstimate(self, steps=1000):
        x = np.linspace(self.xmin, self.xmax, steps)
        y = self._eval_function(x, *self.parameters)
        y = y[np.isfinite(y)]
        return np.max(y) * 1.2 if len(y) > 0 else 1.0

    def Draw(self, option=""):
        import matplotlib.pyplot as plt  # Always import here, unconditionally
        x = np.linspace(self.xmin, self.xmax, 1000)
        y = self._eval_function(x, *self.parameters)

        same_plot = "SAME" in option.upper()

        if not same_plot:
            plt.figure(figsize=(10, 7))

        plt.plot(x, y, 'r-', linewidth=2, label=self.name)

        if not same_plot:
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title(self.name)
            plt.grid(True, alpha=0.3)
            plt.legend()




    def GetChisquare(self):
        return getattr(self, "chi2", 0)

    def GetNDF(self):
        return getattr(self, "ndf", 0)


    def GetProb(self):
        from scipy.stats import chi2 as chi2dist
        if self.ndf > 0:
            return 1.0 - chi2dist.cdf(self.chi2, self.ndf)
        return 0.0



#----------------------------------------------------------------------------------------------#
class TH1F:
    _instances = {}  # Class-level registry for SAME drawing
    
    def __init__(self, name, title, nbins, xmin, xmax):
        self.name = name
        self.title = title
        self.nbins = nbins
        self.xmin = xmin
        self.xmax = xmax
        self.bin_edges = np.linspace(xmin, xmax, nbins + 1)
        self.counts = np.zeros(nbins)
        self.errors = np.zeros(nbins)
        self.fit_function = None
        self.line_color = 1
        self.line_width = 1
        self.fill_color = 0
        self.marker_style = 20
        self.marker_size = 1.0
        
        # Register instance
        TH1F._instances[name] = self
        
    def Fill(self, value, weight=1.0):
        if np.isnan(value) or np.isinf(value):
            return
        bin_index = np.searchsorted(self.bin_edges, value, side='right') - 1
        if 0 <= bin_index < self.nbins:
            self.counts[bin_index] += weight
            self.errors[bin_index] = np.sqrt(self.counts[bin_index])
    
    def Reset(self):
        """Reset histogram contents"""
        self.counts = np.zeros(self.nbins)
        self.errors = np.zeros(self.nbins)
    
    def GetBinContent(self, bin_idx):
        if 1 <= bin_idx <= self.nbins:
            return self.counts[bin_idx - 1]
        return 0
    
    def GetBinError(self, bin_idx):
        if 1 <= bin_idx <= self.nbins:
            return self.errors[bin_idx - 1]
        return 0
    
    def SetLineColor(self, color):
        self.line_color = color
        self._mpl_line_color = ROOTConstants.get_mpl_color(color)

    def SetLineStyle(self, style):
        """Set line style using ROOT constants"""
        self.line_style = style
        self._mpl_line_style = ROOTConstants.get_mpl_linestyle(style)
    
    def SetLineWidth(self, width):
        self.line_width = width
    
    def SetFillColor(self, color):
        self.fill_color = color
        self._mpl_fill_color = ROOTConstants.get_mpl_color(color)
    
    def SetMarkerStyle(self, style):
        self.marker_style = style
        self._mpl_marker = ROOTConstants.get_mpl_marker(style)
    
    def SetMarkerSize(self, size):
        self.marker_size = size
    
    def Fit(self, function, option=""):
        """Enhanced fitting with better error handling"""
        if isinstance(function, TF1):
            bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
            
            # Determine fit range
            if "R" in option.upper():
                mask = (bin_centers >= function.xmin) & (bin_centers <= function.xmax)
            else:
                mask = self.counts > 0
            
            if np.sum(mask) < len(function.parameters):
                print(f"[WARNING] Not enough data points for fitting ({np.sum(mask)} points, {len(function.parameters)} parameters)")
                return
            
            x_data = bin_centers[mask]
            y_data = self.counts[mask]
            
            # Better error estimation
            sigma = np.sqrt(np.maximum(self.counts[mask], 1.0))
            
            try:
                # Try fitting with bounds if parameters seem reasonable
                bounds = ([-np.inf] * len(function.parameters), [np.inf] * len(function.parameters))
                
                popt, pcov = curve_fit(
                    function._eval_function,
                    x_data, y_data,
                    p0=function.parameters,
                    sigma=sigma,
                    absolute_sigma=True,
                    maxfev=10000,
                    bounds=bounds
                )
                
                function.parameters = popt
                function.parameter_errors = np.sqrt(np.diag(pcov))
                self.fit_function = function
                
                # Calculate chi-square
                y_fit = function._eval_function(x_data, *popt)
                chi2 = np.sum(((y_data - y_fit) / sigma)**2)
                ndf = len(x_data) - len(popt)
                function.chi2 = chi2
                function.ndf = ndf
                
                print(f"[INFO] Fit converged: χ²/ndf = {chi2:.2f}/{ndf} = {chi2/ndf:.2f}")
                for i, (param, error) in enumerate(zip(popt, function.parameter_errors)):
                    print(f"       Parameter {i}: {param:.6f} ± {error:.6f}")
                    
            except Exception as e:
                print(f"[ERROR] Fit failed: {e}")



    def Draw(self, option=""):
        """Updated Draw method using enhanced color support"""
        bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        bin_width = self.bin_edges[1] - self.bin_edges[0]
        
        # Use enhanced color mapping
        line_color = getattr(self, '_mpl_line_color', ROOTColors.get_color(self.line_color))
        fill_color = getattr(self, '_mpl_fill_color', line_color)
        line_style = getattr(self, '_mpl_line_style', '-')
        marker = getattr(self, '_mpl_marker', 'o')
        
        same_plot = "SAME" in option.upper()
        
        if not same_plot:
            plt.figure(figsize=(10, 7))
        
        if "E" in option.upper():
            plt.errorbar(bin_centers, self.counts, yerr=self.errors,
                        fmt=marker, color=line_color, markersize=self.marker_size*3,
                        capsize=3, linewidth=self.line_width, 
                        linestyle=line_style, label=self.name)
        elif "HIST" in option.upper() or option == "":
            plt.step(bin_centers, self.counts, where='mid', 
                    color=line_color, linewidth=self.line_width, 
                    linestyle=line_style, label=self.name)
        else:
            alpha = 0.7 if self.fill_color == 0 else 0.9
            plt.bar(bin_centers, self.counts, width=bin_width,
                   align='center', edgecolor=line_color, alpha=alpha,
                   color=fill_color, linewidth=self.line_width,
                   label=self.name)
        
        if not same_plot:
            plt.xlabel("Value")
            plt.ylabel("Counts") 
            plt.title(self.title)
            plt.grid(True, alpha=0.3)
        from ROOT import TCanvas  # import only if ROOT is your wrapper module
        canvas = TCanvas._current_canvas
        if canvas:
            default_filename = f"{self.name}.png"
            canvas.SaveAs(default_filename)
 
###############################################################################

    def FillRandom(self, func_name, nevents=10000):
        from random import uniform

        f = TF1.get_by_name(func_name)
        if f is None:
            try:
                f = TF1.builtin(func_name)
            except Exception as e:
                raise ValueError(f"[FillRandom ERROR] {e}")

        max_y = f.GetMaximumEstimate()

        filled = 0
        attempts = 0
        while filled < nevents and attempts < nevents * 100:
            x_try = uniform(f.xmin, f.xmax)
            y_try = uniform(0, max_y)
            y_val = f._eval_function(x_try, *f.parameters)
            if y_try < y_val:
                self.Fill(x_try)
                filled += 1
            attempts += 1

        print(f"[INFO] FillRandom(): Filled {filled} entries using function '{func_name}'")

###############################################################################

# --- Built-in TF1 factory for known function names like "gaus" ---
    def builtin_tf1_factory(name):
        if name == "gaus":
            f = TF1("gaus", "[0]*np.exp(-0.5*((x-[1])/[2])**2)", -10, 10)
            f.SetParameters(1.0, 0.0, 1.0)  # A, mean, sigma
            return f
        elif name == "expo":
            f = TF1("expo", "[0]*np.exp([1]*x)", 0, 10)
            f.SetParameters(1.0, -1.0)
            return f
    # Add more as needed
        else:
            raise ValueError(f"[ERROR] Unknown built-in function: {name}")




    def Save(self, filename):
        """Save histogram to JSON"""
        data = {
            'name': self.name,
            'title': self.title,
            'nbins': self.nbins,
            'xmin': self.xmin,
            'xmax': self.xmax,
            'counts': self.counts.tolist(),
            'errors': self.errors.tolist(),
            'bin_edges': self.bin_edges.tolist()
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Histogram saved to {filename}")
    
    @classmethod
    def Load(cls, filename):
        """Load histogram from JSON"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        hist = cls(data['name'], data['title'], data['nbins'], data['xmin'], data['xmax'])
        hist.counts = np.array(data['counts'])
        hist.errors = np.array(data['errors'])
        hist.bin_edges = np.array(data['bin_edges'])
        print(f"[INFO] Histogram loaded from {filename}")
        return hist


class TCanvas:
    _current_canvas = None
    _canvas_count = 0  # For auto-naming

    def __init__(self, name=None, title=None, width=800, height=600):
        if name is None:
            TCanvas._canvas_count += 1
            name = f"canvas_{TCanvas._canvas_count}"
        if title is None:
            title = name

        self.name = name
        self.title = title
        self.width = width
        self.height = height
        self.filename = f"{self.name}.png"  # Default save file

        plt.figure(figsize=(width/100, height/100))
        plt.suptitle(title)
        TCanvas._current_canvas = self

    def SaveAs(self, filename=None):
        if filename is None:
            filename = self.filename

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"[+] Canvas saved as: {os.path.abspath(filename)}")

        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                from IPython.display import Image, display
                display(Image(filename))
            except ImportError:
                pass

    def Print(self, filename):
        self.SaveAs(filename)





class TLegend:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.entries = []
        self.border_size = 1
        self.fill_style = 1001
        
    def AddEntry(self, obj, label, option=""):
        self.entries.append((obj, label, option))
    
    def SetBorderSize(self, size):
        self.border_size = size
    
    def SetFillStyle(self, style):
        self.fill_style = style
    
    def Draw(self):
        if self.entries:
            # Use matplotlib legend
            plt.legend(loc='upper right', frameon=self.border_size > 0)

class TLatex:
    def __init__(self):
        self.text_size = 0.04
        self.ndc = False
        
    def SetNDC(self, ndc=True):
        self.ndc = ndc
    
    def SetTextSize(self, size):
        self.text_size = size
    
    def DrawLatex(self, x, y, text):
        # Enhanced ROOT LaTeX to matplotlib conversion
        conversions = {
            '#mu': r'$\mu$', '#sigma': r'$\sigma$', '#pm': r'$\pm$',
            '#alpha': r'$\alpha$', '#beta': r'$\beta$', '#gamma': r'$\gamma$',
            '#delta': r'$\delta$', '#epsilon': r'$\varepsilon$', '#chi': r'$\chi$',
            '#pi': r'$\pi$', '#theta': r'$\theta$', '#phi': r'$\phi$',
            '#lambda': r'$\lambda$', '#nu': r'$\nu$', '#tau': r'$\tau$',
        }
        
        for root_sym, mpl_sym in conversions.items():
            text = text.replace(root_sym, mpl_sym)
        
        fontsize = self.text_size * 300  # Scale appropriately
        
        if self.ndc:
            plt.text(x, y, text, transform=plt.gca().transAxes, fontsize=fontsize)
        else:
            plt.text(x, y, text, fontsize=fontsize)

# Main ROOT module
class ROOT:
    TFile = TFile
    TTree = TTree
    TH1F = TH1F
    TF1 = TF1
    TCanvas = TCanvas
    TLegend = TLegend
    TLatex = TLatex
    gRandom = gRandom
    
    # Add all ROOT constants as class attributes
    kWhite = ROOTConstants.kWhite
    kBlack = ROOTConstants.kBlack
    kGray = ROOTConstants.kGray
    kRed = ROOTConstants.kRed
    kGreen = ROOTConstants.kGreen
    kBlue = ROOTConstants.kBlue
    kYellow = ROOTConstants.kYellow
    kMagenta = ROOTConstants.kMagenta
    kCyan = ROOTConstants.kCyan
    kOrange = ROOTConstants.kOrange
    kSpring = ROOTConstants.kSpring
    kTeal = ROOTConstants.kTeal
    kAzure = ROOTConstants.kAzure
    kViolet = ROOTConstants.kViolet
    kPink = ROOTConstants.kPink
    
    # Line styles
    kSolid = ROOTConstants.kSolid
    kDashed = ROOTConstants.kDashed
    kDotted = ROOTConstants.kDotted
    kDashDotted = ROOTConstants.kDashDotted
    
    # Marker styles
    kDot = ROOTConstants.kDot
    kPlus = ROOTConstants.kPlus
    kStar = ROOTConstants.kStar
    kCircle = ROOTConstants.kCircle
    kMultiply = ROOTConstants.kMultiply
    kFullCircle = ROOTConstants.kFullCircle
    kFullSquare = ROOTConstants.kFullSquare
    kFullTriangleUp = ROOTConstants.kFullTriangleUp
    kFullTriangleDown = ROOTConstants.kFullTriangleDown
    kOpenCircle = ROOTConstants.kOpenCircle
    kOpenSquare = ROOTConstants.kOpenSquare
    kOpenTriangleUp = ROOTConstants.kOpenTriangleUp
    kOpenDiamond = ROOTConstants.kOpenDiamond
    kOpenCross = ROOTConstants.kOpenCross
    kFullStar = ROOTConstants.kFullStar
    kOpenStar = ROOTConstants.kOpenStar

# Module setup
import sys
_root_module = ROOT()
sys.modules['ROOT'] = _root_module

# Also make classes available at module level
for attr_name in dir(_root_module):
    if not attr_name.startswith('_'):
        setattr(sys.modules[__name__], attr_name, getattr(_root_module, attr_name))

print("[INFO] Advanced PyROOT wrapper loaded successfully!")
if UPROOT_AVAILABLE:
    print("[INFO] - Real ROOT file support via uproot")
print("[INFO] - Enhanced selection parsing")  
print("[INFO] - Histogram serialization")
print("[INFO] - Multi-histogram overlays")
print("[INFO] - ROOT color mapping")
print("[INFO] - Complex expression evaluation")
