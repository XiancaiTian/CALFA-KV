"""Frozen calibration splits and native output acceptance for CALFA-KV."""
FIT = [w for w in range(64) if w % 4 != 3]
VALIDATION = [w for w in range(64) if w % 4 == 3]
SPLITS = dict(fit=FIT, validation=VALIDATION,
              fit_gpqa=[w for w in FIT if w < 48], fit_code=[w for w in FIT if w >= 48],
              validation_gpqa=[w for w in VALIDATION if w < 48],
              validation_code=[w for w in VALIDATION if w >= 48],
              all=list(range(64)))


def accepts(current, candidate, parent):
    """Only output reconstruction metrics; all thresholds fixed before fitting."""
    reasons = []
    if not candidate["fit"] <= current["fit"] * (1 - 0.0005):
        reasons.append("fit_improvement_below_0.05_percent")
    for name in ("fit_gpqa", "fit_code"):
        if candidate[name] > current[name] * 1.005 + 1e-12:
            reasons.append(name + "_output_regression")
    if candidate["validation"] > parent["validation"] * 1.005 + 1e-12:
        reasons.append("validation_output_regression")
    for name in ("validation_gpqa", "validation_code"):
        if candidate[name] > parent[name] * 1.01 + 1e-12:
            reasons.append(name + "_output_regression")
    return not reasons, reasons
