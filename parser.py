# parser.py
# -*- coding: utf-8 -*-
"""parser.py

Parser + Wildcard Expansion (x-only)
-----------------------------------

Dieses Modul liefert dir alles, was dein CLI für Custom-Observables braucht:

1) SAFE_FUNCS
   White-List an numpy-vektorisierten Funktionen (aus deinem alten Tool übernommen).

2) ExpressionParserXOnly
   AST-Whitelist-Validator für Ausdrücke, die NUR von x abhängen.
   Erlaubt:
   - Variablen: x1..xN
   - Operatoren: +, -, *, /, ** sowie unary +/ -
   - Funktionsaufrufe: nur direkte Namen aus SAFE_FUNCS (ohne Keywords)
   Verbietet:
   - u-Variablen (u1..)
   - Aliase x/y/u
   - Attribute/Subscripts/Comparisons/Lambdas/...

3) Placeholder-Expansion (Wildcard)
   Pattern wie:
     x           -> x1, x2, ..., xN
     x*x         -> x1*x1, x2*x2, ...
     x*y*z*a     -> alle Kombinationen unterschiedlicher Indizes
     a*a*b       -> ein Index doppelt (Quadrat) + ein anderer Index
   Regel:
   - unterschiedliche Buchstaben => unterschiedliche Indizes
   - wiederholte Buchstaben      => gleicher Index mehrfach
   - max. (#verschiedene Buchstaben) <= nx

4) Dedup/Kanonisierung + Signaturen
   - reine Produkte aus x<i> werden sortiert gespeichert (x2*x1 -> x1*x2)
   - add_dedup(...) überspringt Kandidaten, die schon existieren
   - compute_signatures(...) erzeugt sha256-Signaturen passend zur Feature-Liste
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import math
import re
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np


# ------------------------------------------------------------
# Erlaubte, vektorisierte Funktionen (bitte bei Bedarf erweitern)
# (1:1 aus deinem alten Tool übernommen)
# ------------------------------------------------------------
SAFE_FUNCS: Dict[str, Callable] = {
    # Grundlegende Math
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "abs": np.abs,
    "tanh": np.tanh, "sqrt": np.sqrt,
    "sign": np.sign,

    # Heaviside / Indikatoren
    "heaviside": lambda a: (a >= 0).astype(float),
    "indicator": lambda x: ((-0.05 < x) & (x < 0.05)).astype(int),

    # Glatte Gates / Schalter
    "sigmoid":  lambda x: 1.0/(1.0 + np.exp(-x)),
    "softstep": lambda x, k=50.0: 1.0/(1.0 + np.exp(-k*x)),
    "softsign": lambda x: x/(1.0 + np.abs(x)),
    "smoothabs": lambda x, eps=1e-3: np.sqrt(x*x + eps*eps),

    # Sättigung / Clipping
    "clip": np.clip,
    "clamp": np.clip,
    "saturate": lambda x, limit=1.0: np.clip(x, -limit, limit),
    "deadzone": lambda x, w=0.1: np.where(np.abs(x) <= w, 0.0,
                                           np.sign(x)*(np.abs(x)-w)),

    # Aktivierungen
    "lrelu": lambda x, a=0.01: np.where(x >= 0, x, a*x),
    "elu":   lambda x, a=1.0: np.where(x >= 0, x, a*(np.exp(x)-1.0)),
    "softplus": lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0),
    "swish": lambda x, b=1.0: x/(1.0 + np.exp(-b*x)),
    "gelu":  lambda x: 0.5*x*(1.0 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3))),

    # Fenster / Bump / Gauss
    "gauss": lambda x, mu=0.0, s=1.0: np.exp(-0.5*((x - mu)/s)**2),
    "bump":  lambda x, a=0.0, b=1.0: np.where((x > a) & (x < b),
                                               np.exp(-1.0/((x-a)*(b-x))), 0.0),

    # Periodik / Wrap
    "wrap":  lambda x, period=2*np.pi: ((x + 0.5*period) % period) - 0.5*period,

    # Utility
    "min": np.minimum, "max": np.maximum, "mod": np.mod,
}


# ------------------------------------------------------------
# AST-Whitelist
# ------------------------------------------------------------
ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
ALLOWED_UNOPS = (ast.UAdd, ast.USub)

_XI_RE = re.compile(r"^x(\d+)$")
_XIND_ALL = re.compile(r"\bx(\d+)\b")
_UIND_ALL = re.compile(r"\bu(\d+)\b")


class ExpressionParserXOnly:
    """Sicherer Parser/Validator für Ausdrücke, die nur von x abhängen."""

    def _validate_ast(self, node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            self._validate_ast(node.body)
            return

        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, ALLOWED_BINOPS):
                raise ValueError(f"Operator nicht erlaubt: {type(node.op).__name__}")
            self._validate_ast(node.left)
            self._validate_ast(node.right)
            return

        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, ALLOWED_UNOPS):
                raise ValueError(f"Unärer Operator nicht erlaubt: {type(node.op).__name__}")
            self._validate_ast(node.operand)
            return

        if isinstance(node, ast.Call):
            # nur direkte Funktionsnamen, keine Attribute (np.sin), keine Lambdas
            if not isinstance(node.func, ast.Name):
                raise ValueError("Nur direkte Funktionsnamen erlaubt (z.B. sin(x1)).")
            fname = node.func.id
            if fname not in SAFE_FUNCS:
                raise ValueError(f"Funktion nicht erlaubt: {fname}")
            if node.keywords:
                raise ValueError("Keywords in Funktionsaufrufen sind nicht erlaubt.")
            for arg in node.args:
                self._validate_ast(arg)
            return

        if isinstance(node, ast.Name):
            name = node.id
            if name in ("x", "y", "u"):
                raise ValueError("Aliase (x,y,u) sind verboten. Nutze x1..xN.")
            if name.startswith("u"):
                raise ValueError("u-Variablen sind in x-only Ausdrücken nicht erlaubt.")
            if _XI_RE.fullmatch(name):
                return
            raise ValueError(f"Unbekannte Variable: {name} (erlaubt: x1..xN)")

        if isinstance(node, (ast.Constant, ast.Num)):
            # nur Zahlen
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
                raise ValueError(f"Konstante nicht erlaubt: {node.value!r}")
            return

        # Alles andere blocken: Compare, Attribute, Subscript, Lambda, BoolOp, ...
        raise ValueError(f"Syntaxelement nicht erlaubt: {type(node).__name__}")

    def validate(self, expr: str, *, nx: int) -> None:
        expr = (expr or "").strip()
        if not expr:
            raise ValueError("Leerer Ausdruck.")

        # u komplett verbieten
        if _UIND_ALL.search(expr):
            raise ValueError("u-Variablen sind in x-only Ausdrücken nicht erlaubt.")

        # x-Indexgrenzen prüfen
        x_indices = [int(i) for i in _XIND_ALL.findall(expr)]
        need_x = max(x_indices) if x_indices else 0
        if need_x > nx:
            raise ValueError(f"Ausdruck benötigt x1..x{need_x}, aber nx={nx}.")

        tree = ast.parse(expr, mode="eval")
        self._validate_ast(tree)

    def compile(self, expr: str, *, nx: int) -> Callable[[np.ndarray], np.ndarray]:
        """Optional: liefert vektorisierte f(X)->(N,)"""
        self.validate(expr, nx=nx)

        x_indices = [int(i) for i in _XIND_ALL.findall(expr)]
        need_x = max(x_indices) if x_indices else 0

        def f(X: np.ndarray) -> np.ndarray:
            X_arr = np.asarray(X)
            if X_arr.ndim != 2:
                raise ValueError("X muss 2D sein: (N, nx).")
            if need_x and X_arr.shape[1] < need_x:
                raise ValueError(f"X hat nur {X_arr.shape[1]} Spalten, benötigt: x1..x{need_x}.")

            N = X_arr.shape[0]
            env: Dict[str, object] = dict(SAFE_FUNCS)
            for i in range(1, need_x + 1):
                env[f"x{i}"] = X_arr[:, i - 1]

            val = eval(expr, {"__builtins__": {}}, env)
            val = np.asarray(val)

            if val.ndim == 0:
                return np.full((N,), float(val))

            val = val.reshape(-1)
            if val.shape[0] != N:
                raise ValueError(f"Ausdruck ergab Länge {val.shape[0]}, erwartet {N}.")
            return val

        return f


# ------------------------------------------------------------
# Placeholder/Wildcard Expansion
# ------------------------------------------------------------
# Erlaubt: Buchstabenprodukte wie a*b*c oder x*y*z*a
_PLACEHOLDER_PRODUCT_RE = re.compile(r"^[A-Za-z](\*[A-Za-z])*$")

# Zusätzlich: Wildcards IN beliebigen Ausdrücken (z.B. sin(x)*abs(x)).
# Wir interpretieren JEDE einzelne Buchstaben-Variable (z.B. x, y, a, b, ...)
# als Platzhalter, der auf State-Variablen x1..xN gemappt wird.
# x1, x2, ... bleiben explizit und werden NICHT expandiert.
_SINGLE_LETTER_NAME_RE = re.compile(r"\b([A-Za-z])\b")


def is_placeholder_product(expr: str) -> bool:
    """True nur für Patterns aus EINZELNEN Buchstaben, z.B. 'x*y*z' oder 'a*a*b'."""
    e = (expr or "").replace(" ", "")
    return bool(_PLACEHOLDER_PRODUCT_RE.fullmatch(e))


def expand_placeholder_product(expr: str, *, nx: int, max_terms: int = 50_000) -> List[str]:
    """Expandiert z.B. 'x*y*z*a' für beliebiges nx.

    Regeln:
    - unterschiedliche Buchstaben => unterschiedliche Indizes
    - wiederholte Buchstaben => gleicher Index mehrfach
    - Ausgabe ist kanonisch (sortierte Index-Multiplikitäten), z.B. x1*x1*x3

    Schutz: max_terms (Kombinations-Explosion).
    """
    e = (expr or "").replace(" ", "")
    if not _PLACEHOLDER_PRODUCT_RE.fullmatch(e):
        raise ValueError(f"Kein Placeholder-Produkt: {expr!r}")

    tokens = e.split("*")  # z.B. ["x","y","z","a"]

    # Unique placeholders in order of first appearance
    uniq: List[str] = []
    for t in tokens:
        if t not in uniq:
            uniq.append(t)

    k = len(uniq)
    if k > nx:
        raise ValueError(f"Pattern hat {k} verschiedene Platzhalter, aber nx={nx}.")

    n_terms = math.comb(nx, k)
    if n_terms > max_terms:
        raise ValueError(
            f"Expansion zu groß: C({nx},{k})={n_terms} > max_terms={max_terms}. "
            "Nimm weniger verschiedene Platzhalter oder setze max_terms höher."
        )

    out: List[str] = []
    for combo in itertools.combinations(range(1, nx + 1), k):
        mapping = {uniq[i]: combo[i] for i in range(k)}
        idxs = [mapping[t] for t in tokens]
        idxs.sort()
        out.append("*".join(f"x{i}" for i in idxs))

    return out


# ------------------------------------------------------------
# Wildcards in beliebigen Ausdrücken (x-only)
# ------------------------------------------------------------
# Beispiel:  sin(x)*abs(x)
#  -> [sin(x1)*abs(x1), sin(x2)*abs(x2), ...]
#
# Beispiel:  sin(x) + cos(y)
#  -> Kombinationen ohne Permutations-Duplikate (i<j):
#     sin(x1)+cos(x2), sin(x1)+cos(x3), ..., sin(x_{nx-1})+cos(x_{nx})
#
# Regel:
# - Jeder einzelne Buchstabe (\b[A-Za-z]\b) ist ein Platzhalter für einen Zustand x<i>.
# - Unterschiedliche Buchstaben bekommen unterschiedliche Indizes.
# - Wiederholter Buchstabe => gleicher Index mehrfach.
# - x1..xN (mit Ziffern) sind bereits explizit und werden NICHT expandiert.
# - u-Variablen sind weiterhin komplett verboten (x-only).

_SINGLE_LETTER_RE = re.compile(r"\b([A-Za-z])\b")


def extract_placeholders(expr: str) -> List[str]:
    """Extrahiert einzelne Buchstaben-Placeholders in Auftretens-Reihenfolge."""
    e = (expr or "").strip()
    seen: Set[str] = set()
    out: List[str] = []
    for m in _SINGLE_LETTER_RE.finditer(e):
        name = m.group(1)
        # SAFE_FUNCS sind mehrbuchstabig; trotzdem absichern
        if name in SAFE_FUNCS:
            continue
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def has_placeholders(expr: str) -> bool:
    return len(extract_placeholders(expr)) > 0


def expand_expression_wildcards(expr: str, *, nx: int, max_terms: int = 50_000) -> List[str]:
    """Expandiert Wildcards in beliebigen Ausdrücken.

    Wenn keine Platzhalter vorkommen, wird [expr] zurückgegeben.
    """
    expr = (expr or "").strip()
    if not expr:
        raise ValueError("Leerer Ausdruck.")

    # u weiterhin verboten
    if _UIND_ALL.search(expr):
        raise ValueError("u-Variablen sind in x-only Ausdrücken nicht erlaubt.")

    placeholders = extract_placeholders(expr)
    if not placeholders:
        return [expr]

    # explizite x1..xN bleiben bestehen; Placeholders kommen zusätzlich vor
    k = len(placeholders)
    if k > nx:
        raise ValueError(f"Ausdruck hat {k} verschiedene Platzhalter, aber nx={nx}.")

    n_terms = math.comb(nx, k)
    if n_terms > max_terms:
        raise ValueError(
            f"Expansion zu groß: C({nx},{k})={n_terms} > max_terms={max_terms}. "
            "Nimm weniger verschiedene Platzhalter oder setze max_terms höher."
        )

    out: List[str] = []
    # Kombinationen (i1<i2<...<ik) vermeiden Permutations-Duplikate.
    for combo in itertools.combinations(range(1, nx + 1), k):
        s = expr
        for p, idx in zip(placeholders, combo):
            # Wortgrenzen: ersetzt nur das Identifier-Token, nicht x1
            s = re.sub(rf"\b{re.escape(p)}\b", f"x{idx}", s)
        out.append(s)

    return out


# ------------------------------------------------------------
# Dedup / Kanonisierung
# ------------------------------------------------------------
_XI_TOKEN = re.compile(r"^x(\d+)$")


def canonical_key(expr: str) -> str:
    """Kanonische Darstellung für reine Produkte aus x<i>.

    Beispiele:
    - 'x2*x1'       -> 'x1*x2'
    - 'x3*x1*x3'    -> 'x1*x3*x3'
    - 'exp(x2)*x1'  -> nur whitespace entfernt (nicht umsortiert)
    """
    e = (expr or "").replace(" ", "")
    if "*" in e:
        parts = e.split("*")
        if all(_XI_TOKEN.fullmatch(p) for p in parts):
            idxs = sorted(int(p[1:]) for p in parts)
            return "*".join(f"x{i}" for i in idxs)
    return e


def add_dedup(existing: List[str], candidates: List[str]) -> Tuple[int, int]:
    """Fügt candidates in existing ein, ohne Duplikate.

    Duplikate werden über canonical_key erkannt.
    Returns: (added, skipped)
    """
    keys: Set[str] = set(canonical_key(x) for x in existing)
    added = 0
    skipped = 0

    for c in candidates:
        k = canonical_key(c)
        if k in keys:
            skipped += 1
            continue
        keys.add(k)
        existing.append(k)
        added += 1

    return added, skipped


def compute_signatures(features: List[str]) -> List[str]:
    """Erzeugt sha256-Signaturen passend zu features (in gleicher Reihenfolge)."""
    out: List[str] = []
    for f in features:
        out.append(hashlib.sha256(f.encode("utf-8")).hexdigest())
    return out
