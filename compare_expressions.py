from sympy import *
from typing import List
import re

def compare_multitrees(t1: List[str], t2: List[str]):
    """
    compares trees in the forest by comparing trees with same indices. Constants in expressions are replaced with -1s and ones for negatives and positives.
    returns similarity score, where the score is increased by 1 for each same pair. 
    """
    # for each kind of action
    assert len(t1) == len(t2)
    sim_score = 0
    for i in range(len(t1)):
        e1 = convert_ty_sympy(t1[i])
        e2 = convert_ty_sympy(t2[i])
        sim_score += 1 if e1.equals(e2) else 0
    return sim_score


def convert_ty_sympy(exp: str) -> Expr:
    """
    converts string like this `((x_0/(x_3-(x_5+x_1)))/(((x_0*x_3)*x_3)*((x_4+x_0)/x_0)))` into a sympy expression.
    Each constant is replaced with -1 if it is negative, and with 1 if it is positive
    """
    # Replace negative numbers with -1, positive numbers with 1
    def repl_const(match):
        val = float(match.group(0))
        return "-1" if val < 0 else "1"

    # Replace all numbers (not part of variable names) with -1 or 1
    exp_clean = re.sub(r'(?<![a-zA-Z_])(-?\d+(\.\d+)?)', repl_const, exp)

    # Convert to sympy expression
    expr = sympify(exp_clean)
    return expr