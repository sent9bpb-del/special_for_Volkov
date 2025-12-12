import numpy as np
import matplotlib.pyplot as plt
from math_model_dynamics import hcho_objective, pozitive_hcho_objective

def test_f(x1, x2):
    return x1**2 + x2**2