"""Phase 6a: combine the SSIM score and YOLO detections into a final verdict.

Decision logic
--------------
The pipeline has two independent signals:

    SSIM score    — measures structural similarity to the reference board.
                    Low SSIM means the board looks globally different (wrong
                    board, severe warping, or many defects at once).

    YOLO detections — spot specific defects even when global SSIM is fine
                      (e.g. a single cold joint on an otherwise perfect board).

Verdict rules (applied in order):
    1. SSIM < ssim_flag (0.80)        → FAIL  (structurally too different)
    2. SSIM ≥ ssim_pass (0.85)
       AND no defects found           → PASS
    3. SSIM ≥ ssim_pass (0.85)
       BUT defects found              → FAIL
    4. ssim_flag ≤ SSIM < ssim_pass  → FLAG  (manual review — borderline board)

Usage
-----
    from scripts.verdict import compute_verdict

    result = compute_verdict(ssim_score=0.91, detections=[])
    # {'verdict': 'PASS', 'reason': 'SSIM=0.9100 ≥ 0.85; no defects detected.'}
"""

from scripts.config import SSIM_FLAG_THRESHOLD, SSIM_PASS_THRESHOLD


def compute_verdict(
    ssim_score: float,
    detections: list,
    ssim_pass: float = SSIM_PASS_THRESHOLD,
    ssim_flag: float = SSIM_FLAG_THRESHOLD,
) -> dict:
    """Produce a final PASS / FAIL / FLAG verdict from the two pipeline signals.

    Parameters
    ----------
    ssim_score  : float — structural similarity score (0.0–1.0)
    detections  : list  — output of detect_defects(); may be empty
    ssim_pass   : float — SSIM above which a board is a PASS candidate (default 0.85)
    ssim_flag   : float — SSIM below which a board is an immediate FAIL (default 0.80)

    Returns
    -------
    dict with keys:
        verdict     : str  — 'PASS', 'FAIL', or 'FLAG'
        reason      : str  — human-readable explanation
        ssim_score  : float
        num_defects : int
        defects     : list[str] — e.g. ['solder_bridge(0.92)', 'cold_joint(0.87)']
    """
    if not isinstance(ssim_score, (int, float)):
        raise TypeError(f"ssim_score must be a number, got {type(ssim_score).__name__}")
    if not isinstance(detections, list):
        raise TypeError(f"detections must be a list, got {type(detections).__name__}")

    num_defects = len(detections)
    defect_tags = [
        f"{d['class_name']}({d['confidence']:.2f})" for d in detections
    ]

    # Rule 1: structural failure — SSIM too low regardless of YOLO
    if ssim_score < ssim_flag:
        reason = (
            f"SSIM={ssim_score:.4f} < {ssim_flag} — board is structurally "
            "too different from reference."
        )
        verdict = "FAIL"

    # Rule 4: borderline — manual review needed
    elif ssim_score < ssim_pass:
        reason = (
            f"SSIM={ssim_score:.4f} is in the review band "
            f"[{ssim_flag}, {ssim_pass}) — manual inspection required."
        )
        if num_defects:
            reason += f" {num_defects} defect(s) also detected: {', '.join(defect_tags)}."
        verdict = "FLAG"

    # Rule 3: good SSIM but defects found
    elif num_defects > 0:
        reason = (
            f"SSIM={ssim_score:.4f} ≥ {ssim_pass} but "
            f"{num_defects} defect(s) detected: {', '.join(defect_tags)}."
        )
        verdict = "FAIL"

    # Rule 2: all clear
    else:
        reason = f"SSIM={ssim_score:.4f} ≥ {ssim_pass}; no defects detected."
        verdict = "PASS"

    return {
        "verdict":     verdict,
        "reason":      reason,
        "ssim_score":  ssim_score,
        "num_defects": num_defects,
        "defects":     defect_tags,
    }
