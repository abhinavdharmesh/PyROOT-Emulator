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
import matplotlib
# forcing interactive backend for popup windows
# REPLACE the setup_interactive_matplotlib function with this:
def setup_interactive_matplotlib():
    """Setup matplotlib for interactive display"""
    import matplotlib
    import matplotlib.pyplot as plt
    
    # Don't force backend change if already set
    current_backend = matplotlib.get_backend()
    
    if 'Agg' in current_backend and current_backend != 'Qt5Agg':
        try:
            matplotlib.use('Qt5Agg')
            print("[INFO] Switched to Qt5Agg backend for interactive plots")
        except ImportError:
            try:
                matplotlib.use('TkAgg')
                print("[INFO] Switched to TkAgg backend for interactive plots") 
            except ImportError:
                print("[WARNING] Keeping non-interactive backend")
    
    plt.ion()  # Turn on interactive mode

# Call setup function
setup_interactive_matplotlib()

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


class TMath:
    """ROOT TMath mathematical functions"""
    
    # Mathematical constants
    Pi = np.pi
    E = np.e
    Ln10 = np.log(10)
    LogE = np.log10(np.e)
    C = 2.99792458e8  # Speed of light
    
    @staticmethod
    def Abs(x):
        return np.abs(x)
    
    @staticmethod
    def Sqrt(x):
        return np.sqrt(x)
    
    @staticmethod
    def Power(x, y):
        return np.power(x, y)
    
    @staticmethod
    def Log(x):
        return np.log(x)
    
    @staticmethod
    def Log10(x):
        return np.log10(x)
    
    @staticmethod
    def Exp(x):
        return np.exp(x)
    
    @staticmethod
    def Sin(x):
        return np.sin(x)
    
    @staticmethod
    def Cos(x):
        return np.cos(x)
    
    @staticmethod
    def Tan(x):
        return np.tan(x)
    
    @staticmethod
    def ASin(x):
        return np.arcsin(x)
    
    @staticmethod
    def ACos(x):
        return np.arccos(x)
    
    @staticmethod
    def ATan(x):
        return np.arctan(x)
    
    @staticmethod
    def ATan2(y, x):
        return np.arctan2(y, x)
    
    @staticmethod
    def Hypot(x, y):
        return np.hypot(x, y)
    
    @staticmethod
    def Min(a, b):
        return np.minimum(a, b)
    
    @staticmethod
    def Max(a, b):
        return np.maximum(a, b)
    
    @staticmethod
    def Range(lb, ub, x):
        """Constrain x to be between lb and ub"""
        return np.clip(x, lb, ub)
    
    @staticmethod
    def Sign(a, b):
        """Return |a| with sign of b"""
        return np.copysign(np.abs(a), b)
    
    # Statistical functions
    @staticmethod
    def Prob(chi2, ndf):
        """Probability for chi-square"""
        from scipy.stats import chi2 as chi2dist
        if ndf > 0:
            return 1.0 - chi2dist.cdf(chi2, ndf)
        return 0.0
    
    @staticmethod
    def Gaus(x, mean=0, sigma=1, norm=False):
        """Gaussian function"""
        result = np.exp(-0.5 * ((x - mean) / sigma) ** 2)
        if norm:
            result /= (sigma * np.sqrt(2 * np.pi))
        return result
    
    @staticmethod
    def Landau(x, mpv=0, sigma=1, norm=False):
        """Landau distribution (approximation)"""
        # Simplified Landau approximation
        z = (x - mpv) / sigma
        result = np.exp(-z - np.exp(-z))
        if norm:
            result /= sigma
        return result
    
    @staticmethod
    def Poisson(x, par):
        """Poisson probability"""
        from scipy.stats import poisson
        return poisson.pmf(x, par)
    
    @staticmethod
    def Gamma(z):
        """Gamma function"""
        from scipy.special import gamma
        return gamma(z)
    
    @staticmethod
    def LnGamma(z):
        """Log of Gamma function"""
        from scipy.special import loggamma
        return loggamma(z)
    
    @staticmethod
    def Factorial(n):
        """Factorial function"""
        from scipy.special import factorial
        return factorial(n)
    
    @staticmethod
    def Binomial(n, k):
        """Binomial coefficient"""
        from scipy.special import comb
        return comb(n, k)
    
    @staticmethod
    def BetaIncomplete(a, b, x):
        """Incomplete beta function"""
        from scipy.special import betainc
        return betainc(a, b, x)
    
    @staticmethod
    def IsNaN(x):
        return np.isnan(x)
    
    @staticmethod
    def IsInf(x):
        return np.isinf(x)
    
    @staticmethod
    def Finite(x):
        return np.isfinite(x)
    
    @staticmethod
    def Floor(x):
        return np.floor(x)
    
    @staticmethod
    def Ceil(x):
        return np.ceil(x)
    
    @staticmethod
    def Nint(x):
        """Nearest integer"""
        return np.round(x).astype(int)
    
    @staticmethod
    def Sort(n, a, index=None, down=False):
        """Sort array"""
        if index is not None:
            idx = np.argsort(a[:n])
            if down:
                idx = idx[::-1]
            index[:n] = idx
        else:
            return np.sort(a[:n])[::-1] if down else np.sort(a[:n])

        
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
    def get_mpl_color(cls, root_color, alpha=1.0):
        """Convert ROOT color constants to matplotlib RGBA colors"""
        base_color = {
            cls.kWhite: 'white', cls.kBlack: 'black', cls.kGray: 'gray',
            cls.kRed: 'red', cls.kGreen: 'green', cls.kBlue: 'blue',
            cls.kYellow: 'yellow', cls.kMagenta: 'magenta', cls.kCyan: 'cyan',
            cls.kOrange: 'orange', cls.kSpring: 'springgreen', cls.kTeal: 'teal',
            cls.kAzure: 'azure', cls.kViolet: 'violet', cls.kPink: 'pink',
            cls.kBlue + 1: 'darkblue', cls.kBlue - 1: 'lightblue',
            cls.kRed + 1: 'darkred', cls.kRed - 1: 'lightcoral',
            cls.kGreen + 1: 'darkgreen', cls.kGreen - 1: 'lightgreen',
            cls.kAzure - 9: 'lightcyan', cls.kAzure + 1: 'darkturquoise',
            cls.kOrange + 1: 'darkorange', cls.kOrange - 1: 'moccasin',
        }.get(root_color, 'black')

    # Convert to RGBA using matplotlib
        import matplotlib.colors as mcolors
        rgba = mcolors.to_rgba(base_color, alpha)
        return rgba  # → (r, g, b, alpha)

    
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

    @staticmethod
    def builtin(name, xmin=-5, xmax=5):
        """
        Mimics ROOT's built-in named TF1 functions (e.g., 'gaus', 'expo').
        Returns a TF1 instance with default parameters.
        """
        if name == "gaus":
            f = TF1("gaus", "[0]*np.exp(-0.5*((x-[1])/[2])**2)", xmin, xmax)
            f.SetParameters(1.0, 0.0, 1.0)  # A, mean, sigma
        elif name == "expo":
            f = TF1("expo", "[0]*np.exp([1]*x)", xmin, xmax)
            f.SetParameters(1.0, -1.0)
        else:
            raise ValueError(f"[TF1 builtin] Unknown function name: '{name}'")

        # Optional: register function to internal registry if needed
        f.register()  # Only if your TF1 class has a registry system
        return f



    def _translate_formula(self, formula):
    # Replace parameters [0], [1], ... → p[0], p[1], ...
        for i in range(10):
            formula = formula.replace(f'[{i}]', f'p[{i}]')

    # Replace function names not already starting with np.
        replacements = {
            r'(?<!\.)\bexp\b': 'np.exp',
            r'(?<!\.)\bsqrt\b': 'np.sqrt',
            r'(?<!\.)\bpow\b': 'np.power',
            r'(?<!\.)\bsin\b': 'np.sin',
            r'(?<!\.)\bcos\b': 'np.cos',
            r'(?<!\.)\btan\b': 'np.tan',
            r'(?<!\.)\blog\b': 'np.log',
            r'(?<!\.)\babs\b': 'np.abs',
            r'\bpi\b': 'np.pi',
            r'TMath::Pi\(\)': 'np.pi',
        }

        for pattern, replacement in replacements.items():
            formula = re.sub(pattern, replacement, formula)

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
        try:
            expr = self.formula
        # Don't replace [i] here since it's already done in _translate_formula
            code = f"lambda x, p: {expr}"
            self._compiled = eval(code, {"np": np})
            self._compiled_valid = True
        except Exception as e:
            print(f"[TF1 compile error] {e}")
            self._compiled = None
            self._compiled_valid = False

    def _eval_function(self, x, *params):
        if not self._compiled:
            self._compile()
        try:
        # Ensure params is a list/array, not a tuple
            if len(params) == 1 and isinstance(params[0], (list, tuple, np.ndarray)):
                params = params[0]
            return self._compiled(x, list(params))
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


    def SetLineColor(self, color):
        self.line_color = color
    def SetLineWidth(self, width):
        self.line_width = width

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
        self.fill_alpha = 1.0  # default opaque
        self.marker_style = 20
        self.marker_size = 1.0
        
        # Register instance
        TH1F._instances[name] = self
 

#--#
    def GetFunction(self, name=None):
        """Return the last fitted TF1 object, mimicking ROOT's GetFunction('name')"""
        if self.fit_function is None:
            raise AttributeError("No function has been fitted yet.")
        if name is None or self.fit_function.name == name:
            return self.fit_function
        else:
            raise AttributeError(f"No fit function named '{name}' found.")

#--#
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

    def SetFillColorAlpha(self, color, alpha):
        self.fill_color = color
        self.fill_alpha = alpha
        self._mpl_fill_color = ROOTConstants.get_mpl_color(color, alpha)

    
    def SetMarkerStyle(self, style):
        self.marker_style = style
        self._mpl_marker = ROOTConstants.get_mpl_marker(style)
    
    def SetMarkerSize(self, size):
        self.marker_size = size

    def SetMarkerColor(self, color):
        self.marker_color = color
        self._mpl_marker_color = ROOTConstants.get_mpl_color(color)
#-------------------------------------------------------------#
    def Fit(self, function, option=""):
        """Enhanced fitting with better error handling. Accepts TF1 or function name string."""
    
        # Add this to support Fit("gaus")
        if isinstance(function, str):
            try:
                function = TF1.builtin(function, self.xmin, self.xmax)
            except Exception as e:
                print(f"[Fit ERROR] Could not parse function name '{function}': {e}")
                return

        # --- Auto set parameters from histogram stats (like ROOT does) ---
            if function.name == "gaus":
                amp = np.max(self.counts)
                mean = self.GetMean()
                sigma = self.GetRMS()
                function.SetParameters(amp, mean, sigma)
            elif function.name == "expo":
                amp = np.max(self.counts)
                slope = -1.0  # fallback guess
                function.SetParameters(amp, slope)

        if not isinstance(function, TF1):
            raise TypeError(f"[Fit ERROR] Expected TF1 or function name string, got: {type(function)}")

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
        sigma = np.sqrt(np.maximum(self.counts[mask], 1.0))

        try:
            # Set reasonable bounds for Gaussian fits
            if function.name == "gaus":
                lower_bounds = [0, self.xmin, 0.001]  # amplitude > 0, mean in range, sigma > 0
                upper_bounds = [np.inf, self.xmax, (self.xmax - self.xmin)]
                bounds = (lower_bounds[:len(function.parameters)], upper_bounds[:len(function.parameters)])
            else:
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

            y_fit = function._eval_function(x_data, *popt)
            chi2 = np.sum(((y_data - y_fit) / sigma) ** 2)
            ndf = len(x_data) - len(popt)
            function.chi2 = chi2
            function.ndf = ndf

            print(f"[INFO] Fit converged: χ²/ndf = {chi2:.2f}/{ndf} = {chi2/ndf:.2f}")
            for i, (param, error) in enumerate(zip(popt, function.parameter_errors)):
                print(f"       Parameter {i}: {param:.6f} ± {error:.6f}")

        except Exception as e:
            print(f"[ERROR] Fit failed: {e}")
#----------------------------------------------------------------#

    def Draw(self, option=""):
        """
        Draw the histogram using matplotlib.
        Supports ROOT-like options: "HIST", "E", "SAME", "LOG".
        """
        import matplotlib.pyplot as plt

        bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        bin_width = self.bin_edges[1] - self.bin_edges[0]

        # Get styling
        line_color = getattr(self, '_mpl_line_color', ROOTColors.get_color(self.line_color))
        fill_color = getattr(self, '_mpl_fill_color', line_color)
        line_style = getattr(self, '_mpl_line_style', '-')
        marker = getattr(self, '_mpl_marker', 'o')
    
        same_plot = "SAME" in option.upper()
        error_plot = "E" in option.upper()
        log_plot = "LOG" in option.upper()

        if not same_plot:
            plt.figure(figsize=(10, 7))

        if error_plot:
    # Get marker color (add this line)
            marker_color = getattr(self, '_mpl_marker_color', line_color)
    
            plt.errorbar(
                bin_centers,
                self.counts,
                yerr=self.errors,
                fmt=marker,
                color=marker_color,  # Use marker_color instead of line_color
                markersize=self.marker_size * 3,
                capsize=3,
                linewidth=self.line_width,
                linestyle=line_style,
                label=self.name
            )
        elif "HIST" in option.upper() or option.strip() == "":
            plt.step(
                bin_centers,
                self.counts,
                where="mid",
                color=line_color,
                linewidth=self.line_width,
                linestyle=line_style,
                label=self.name
            )
        else:
            alpha = 0.7 if self.fill_color == 0 else 0.9
            plt.bar(
                bin_centers,
                self.counts,
                width=bin_width,
                align='center',
                edgecolor=line_color,
                alpha=alpha,
                color=fill_color,
                linewidth=self.line_width,
                label=self.name
            )

        if log_plot:
            plt.yscale("log")

        if not same_plot:
            plt.xlabel("Value")
            plt.ylabel("Counts")
            plt.title(self.title)
            plt.grid(True, alpha=0.3)
            plt.legend()

    # Save if using canvas
        from ROOT import TCanvas
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


    def GetEntries(self):
        return float(np.sum(self.counts))
    def GetMean(self):
        bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        total = np.sum(self.counts)
        if total == 0:
            return 0.0
        return float(np.sum(bin_centers * self.counts) / total)

    def GetRMS(self):
        mean = self.GetMean()
        bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        total = np.sum(self.counts)
        if total == 0:
            return 0.0
        variance = np.sum(((bin_centers - mean) ** 2) * self.counts) / total
        return float(np.sqrt(variance))
    def Integral(self, bin1=None, bin2=None):

        if bin1 is None and bin2 is None:
        # Integrate entire histogram
            return float(np.sum(self.counts))
    
        if bin1 is None:
            bin1 = 1
        if bin2 is None:
            bin2 = self.nbins
    
    # Convert to 0-based indexing and ensure valid range
        start_idx = max(0, bin1 - 1)
        end_idx = min(self.nbins, bin2)
    
        if start_idx >= end_idx:
            return 0.0
    
        return float(np.sum(self.counts[start_idx:end_idx]))

    def IntegralAndError(self, bin1=None, bin2=None):

        if bin1 is None and bin2 is None:
            integral = float(np.sum(self.counts))
            error = float(np.sqrt(np.sum(self.errors**2)))
        else:
            if bin1 is None:
                bin1 = 1
            if bin2 is None:
                bin2 = self.nbins
            
            start_idx = max(0, bin1 - 1)
            end_idx = min(self.nbins, bin2)
        
            if start_idx >= end_idx:
                return 0.0, 0.0
            
            integral = float(np.sum(self.counts[start_idx:end_idx]))
            error = float(np.sqrt(np.sum(self.errors[start_idx:end_idx]**2)))
    
        return integral, error


    def GetMaximum(self):
        """Get maximum bin content"""
        return float(np.max(self.counts)) if len(self.counts) > 0 else 0.0

    def GetMinimum(self):
        """Get minimum bin content"""
        return float(np.min(self.counts)) if len(self.counts) > 0 else 0.0

    def GetMaximumBin(self):
        """Get bin number with maximum content (1-based)"""
        if len(self.counts) == 0:
            return 0
        return int(np.argmax(self.counts)) + 1

    def GetMinimumBin(self):
        """Get bin number with minimum content (1-based)"""
        if len(self.counts) == 0:
            return 0
        return int(np.argmin(self.counts)) + 1

    def Scale(self, factor):
        """Scale histogram by a factor"""
        self.counts *= factor
        self.errors *= factor

    def Add(self, other_hist, scale=1.0):
        """Add another histogram to this one"""
        if isinstance(other_hist, TH1F):
            if len(self.counts) == len(other_hist.counts):
                self.counts += scale * other_hist.counts
            # Add errors in quadrature
                self.errors = np.sqrt(self.errors**2 + (scale * other_hist.errors)**2)
            else:
                print("[WARNING] Cannot add histograms with different binning")
        else:
            print("[WARNING] Can only add TH1F objects")

    def FindBin(self, x):
        """Find bin number for given x value (1-based)"""
        if x < self.xmin or x > self.xmax:
            return 0  # Underflow/overflow
    
        bin_index = np.searchsorted(self.bin_edges, x, side='right') - 1
        return bin_index + 1  # Convert to 1-based

    def GetBinCenter(self, bin_num):
        """Get center of bin (1-based bin numbering)"""
        if 1 <= bin_num <= self.nbins:
            idx = bin_num - 1
            return 0.5 * (self.bin_edges[idx] + self.bin_edges[idx + 1])
        return 0.0

    def GetBinWidth(self, bin_num=1):
        """Get width of bin (all bins have same width in TH1F)"""
        return (self.xmax - self.xmin) / self.nbins


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





class TGraph:
    """ROOT TGraph implementation"""
    
    def __init__(self, n=0, x=None, y=None):
        self.n = n
        self.x = np.array(x) if x is not None else np.array([])
        self.y = np.array(y) if y is not None else np.array([])
        self.title = ""
        self.name = f"Graph_{id(self)}"
        
        # Style attributes
        self.line_color = 1
        self.line_width = 1
        self.line_style = 1
        self.fill_color = 0
        self.marker_color = 1  
        self.marker_style = 20
        self.marker_size = 1.0

#------------------#
#------------------#        
    def SetPoint(self, i, x, y):
        """Set point i to (x,y)"""
        if i >= len(self.x):
            # Extend arrays
            new_size = i + 1
            new_x = np.zeros(new_size)
            new_y = np.zeros(new_size)
            new_x[:len(self.x)] = self.x
            new_y[:len(self.y)] = self.y
            self.x = new_x
            self.y = new_y
            self.n = new_size
        self.x[i] = x
        self.y[i] = y
    
    def GetPoint(self, i):
        """Get point i"""
        if 0 <= i < len(self.x):
            return self.x[i], self.y[i]
        return 0, 0
    
    def GetN(self):
        return len(self.x)
    
    def SetTitle(self, title):
        self.title = title
    
    def SetName(self, name):
        self.name = name
    
    def SetLineColor(self, color):
        self.line_color = color
    
    def SetLineWidth(self, width):
        self.line_width = width
    
    def SetLineStyle(self, style):
        self.line_style = style
    
    def SetMarkerColor(self, color):
        self.marker_color = color
        self._mpl_marker_color = ROOTConstants.get_mpl_color(color)
    
    def SetMarkerStyle(self, style):
        self.marker_style = style
    
    def SetMarkerSize(self, size):
        self.marker_size = size
    
    def Draw(self, option=""):
        """Draw the graph"""
        import matplotlib.pyplot as plt
        
        line_color = ROOTConstants.get_mpl_color(self.line_color)
        marker_color = ROOTConstants.get_mpl_color(self.marker_color)
        marker = ROOTConstants.get_mpl_marker(self.marker_style)
        linestyle = ROOTConstants.get_mpl_linestyle(self.line_style)
        
        same_plot = "SAME" in option.upper()
        
        if not same_plot:
            plt.figure(figsize=(10, 7))
        
        if "L" in option.upper():  # Line only
            plt.plot(self.x, self.y, color=line_color, linewidth=self.line_width,
                    linestyle=linestyle, label=self.name)
        elif "P" in option.upper():  # Points only
            plt.scatter(self.x, self.y, c=marker_color, marker=marker,
                       s=self.marker_size*20, label=self.name)
        else:  # Both line and points
            plt.plot(self.x, self.y, color=line_color, linewidth=self.line_width,
                    linestyle=linestyle, marker=marker, markersize=self.marker_size*3,
                    markerfacecolor=marker_color, label=self.name)
        
        if not same_plot:
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(self.title)
            plt.grid(True, alpha=0.3)
            plt.legend()

class TGraphErrors(TGraph):
    """ROOT TGraphErrors implementation"""
    
    def __init__(self, n=0, x=None, y=None, ex=None, ey=None):
        super().__init__(n, x, y)
        self.ex = np.array(ex) if ex is not None else np.zeros_like(self.x)
        self.ey = np.array(ey) if ey is not None else np.zeros_like(self.y)
    
    def SetPointError(self, i, ex, ey):
        """Set error for point i"""
        if i < len(self.ex):
            self.ex[i] = ex
            self.ey[i] = ey
    
    def GetErrorX(self, i):
        if 0 <= i < len(self.ex):
            return self.ex[i]
        return 0
    
    def GetErrorY(self, i):
        if 0 <= i < len(self.ey):
            return self.ey[i]
        return 0
    
    def Draw(self, option=""):
        """Draw graph with error bars"""
        import matplotlib.pyplot as plt
        
        line_color = ROOTConstants.get_mpl_color(self.line_color)
        marker = ROOTConstants.get_mpl_marker(self.marker_style)
        
        same_plot = "SAME" in option.upper()
        
        if not same_plot:
            plt.figure(figsize=(10, 7))
        
        plt.errorbar(self.x, self.y, xerr=self.ex, yerr=self.ey,
                    fmt=marker, color=line_color, markersize=self.marker_size*3,
                    linewidth=self.line_width, capsize=3, label=self.name)
        
        if not same_plot:
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(self.title)
            plt.grid(True, alpha=0.3)
            plt.legend()


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
        plt.close('all')
        plt.figure(figsize=(width/100, height/100))
        plt.suptitle(title)
        TCanvas._current_canvas = self

    def Show(self, block=False):
        """Display canvas in popup window without saving"""
        plt.tight_layout()
        plt.show(block=block)
        if not block:
            plt.pause(0.1)  # Ensure window appears

    def SaveAs(self, filename=None):
        if filename is None:
            filename = self.filename

        plt.tight_layout()
    
    # Save the file first
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"[+] Canvas saved as: {os.path.abspath(filename)}")

    # Show interactive plot ONLY if backend supports it
        import matplotlib
        backend = matplotlib.get_backend()
    
        if backend in ['Qt5Agg', 'TkAgg', 'GTK3Agg', 'Qt4Agg']:
        # Don't create new figure, just show current one
            plt.show(block=False)
            plt.pause(0.1)
        else:
            try:
                plt.show(block=False)
            except:
                print("[INFO] Interactive display not available")

    # Remove IPython display code that might cause second window
    # (Comment out or remove the IPython.display.Image section)




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
# Add to ROOT class
    TMath = TMath
    TGraph = TGraph
    TGraphErrors = TGraphErrors    
    
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
