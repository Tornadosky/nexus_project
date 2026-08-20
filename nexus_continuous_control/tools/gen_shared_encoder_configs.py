"""Generate the shared-encoder campaign configs from their bases.

Each new config is derived TEXTUALLY from an existing committed config with a
minimal, targeted edit, so that only the intended keys can possibly differ.
Run from the repo root; verify afterwards with _ydiff_campaign.py.
"""
import pathlib
import re

C = pathlib.Path("configs")


def read(p):
    return (C / p).read_text()


def set_key(text, key, val):
    """Replace an existing top-level key value, preserving any trailing comment."""
    pat = re.compile(r"^(%s:)([^\n#]*)(#.*)?$" % re.escape(key), re.M)
    assert pat.search(text), "key %s not found" % key

    def rep(m):
        tail = ("        " + m.group(3)) if m.group(3) else ""
        return "%s %s%s" % (m.group(1), val, tail)

    return pat.sub(rep, text, count=1)


def insert_after(text, anchor_key, newline):
    pat = re.compile(r"^%s:.*$" % re.escape(anchor_key), re.M)
    m = pat.search(text)
    assert m, "anchor %s not found" % anchor_key
    return text[: m.end()] + "\n" + newline + text[m.end():]


def strip_header(text):
    """Drop the leading comment block so we can put our own provenance on top."""
    lines = text.split("\n")
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
        i += 1
    return "\n".join(lines[i:])


def write(name, header, body):
    (C / name).write_text(header.rstrip() + "\n" + body.lstrip("\n"))
    print("wrote configs/%s" % name)


# --------------- TIER 0: the missing aux-off control at MDI=4 ----------------
T0_HDR = """# %(env)s RGB -- TIER 0: THE MISSING CONTROL (2026-08-19).
#
# %(base)s with RGB_AUX_STATE_COEF: 0.0 and NOTHING ELSE changed.
#
# WHY THIS EXISTS. The 2026-08-18 "fix" changed THREE things at once:
#   (a) RGB_AUX_STATE_COEF 0.0 -> 1.0   dense pixel->state supervision
#   (b) META_DECISION_INTERVAL 1 -> 4   stop the meta deciding every step
#   (c) LR 1e-4 -> 3e-4                 (cartpole only; walker was already 3e-4)
# and the resulting arm SEES. No aux-off control at MDI=4 was ever run, so it is
# not known which of the three did the work. This arm isolates (a): keep the
# MDI and LR of the committed fix, turn the aux crutch OFF.
#
# READ IT AS:
#   still SEES (median pixel drop > 0.30) -> the aux loss was NEVER load-bearing;
#       MDI (and/or LR) alone cured the blindness. The shared-encoder hypothesis
#       then loses its premise, because there is no aux crutch left to remove.
#   BLIND (drop ~ 0)                      -> the aux loss really is doing the work,
#       and "can sharing replace it?" is a well-posed question.
#
# It is also the correct denominator for the shared-encoder arm: the only
# difference to %(shared)s is RGB_SHARED_ENCODER.
#
# INTEGRITY: train/rgb/aux_state_loss must be EXACTLY 0 for the whole run.
"""

t0c = set_key(read("cartpole_balance_nesy_rgb_aux.yaml"), "RGB_AUX_STATE_COEF", "0.0")
t0c = t0c.replace("# --- the fix ---", "# --- the fix, WITHOUT the aux crutch ---")
write(
    "cartpole_balance_nesy_rgb_noaux.yaml",
    T0_HDR
    % dict(
        env="CartpoleBalance",
        base="cartpole_balance_nesy_rgb_aux.yaml",
        shared="cartpole_balance_nesy_rgb_shared_noaux.yaml",
    ),
    strip_header(t0c),
)

t0w = set_key(read("walker_walk_nesy_rgb_aux.yaml"), "RGB_AUX_STATE_COEF", "0.0")
t0w = t0w.replace("# --- the fix ---", "# --- the fix, WITHOUT the aux crutch ---")
write(
    "walker_walk_nesy_rgb_noaux.yaml",
    T0_HDR
    % dict(
        env="WalkerWalk",
        base="walker_walk_nesy_rgb_aux.yaml",
        shared="walker_walk_nesy_rgb_shared_noaux.yaml",
    ),
    strip_header(t0w),
)

# ------------- TIER 1a: shared encoder on top of the Tier-0 control ----------
T1_SHARED_HDR = """# WalkerWalk RGB: LEVER A ALONE -- one shared CNN trunk, no aux crutch.
#
# THE LOAD-BEARING ARM for walker. It is walker_walk_nesy_rgb_noaux.yaml
# (the Tier-0 control) plus RGB_SHARED_ENCODER: true, and nothing else, so the
# ONE-FLAG comparison against that file isolates the effect of sharing.
#
# THE HYPOTHESIS. By default each of the N skill actors carries its OWN CNN (the
# whole VisionSkillActor is vmapped over N init keys), so each private encoder
# receives only its 1/N share of the deterministic policy gradient through the
# privileged critic -- the only signal it gets. One shared trunk receives the
# SUMMED gradient of all N heads instead. If blindness was only ever about
# gradient MAGNITUDE and not about the KIND of signal, sharing alone should cure
# it and the privileged aux regression target can be dropped entirely.
#
# WHY WALKER IS THE TESTBED. Its blind baseline has the sharpest WITHIN-RUN
# contrast we have: three of four skills produce 0.000% action spread across
# completely different frames (bit-identical, tanh-saturated output) while the
# fourth, energy_efficient, produces 33.85%. One run therefore contains both the
# failure and the success case, so a fix cannot be confused with between-run
# variance. THE PREDICTION: with a shared trunk the one skill that learned to
# see should drag the other three off 0.000%, landing all four in the same
# responsiveness band -- which is what the aux fix achieved (38.1/36.9/32.7/34.1%).
#
# INTEGRITY: train/rgb/aux_state_loss must be EXACTLY 0 for the whole run.
"""
t1w = insert_after(
    t0w,
    "RGB_AUX_STATE_COEF",
    "RGB_SHARED_ENCODER: true       # LEVER A: one CNN trunk, N small action heads",
)
write("walker_walk_nesy_rgb_shared_noaux.yaml", T1_SHARED_HDR, strip_header(t1w))

# ------ TIER 1b: sharing alone on the BLIND anchor config (MDI stays 1) ------
T1_ONLY_HDR = """# %(env)s: the BLIND ANCHOR config plus RGB_SHARED_ENCODER, nothing else.
#
# %(base)s + RGB_SHARED_ENCODER: true. A strict ONE-FLAG test against the
# committed blind anchor (%(anchor)s), which is the same
# config run through the same ablation harness.
#
# READ THIS ARM CAREFULLY -- "SHARING ALONE, SHORTCUT INTACT".
# META_DECISION_INTERVAL stays at its default of 1 here, so the meta-Q still
# re-picks a skill EVERY step and can compensate for a blind actor by itself.
# That shortcut is exactly what META_DECISION_INTERVAL: 4 removes in the Tier-0
# and Tier-1a arms. Therefore:
#   SEES  -> sharing cures blindness even with the meta shortcut available:
#            a strong result.
#   BLIND -> this is NOT evidence against sharing. It means removing the meta
#            shortcut is NECESSARY for sharing to bite, and the Tier-1a arm
#            (MDI=4) is the one that answers the hypothesis.
#
# INTEGRITY: no aux loss is configured here, so train/rgb/aux_state_loss must be
# EXACTLY 0 for the whole run.
"""

# cartpole blind anchor + shared
cb = read("cartpole_balance_nesy_rgb.yaml")
cb = insert_after(
    cb, "RGB_AUG_PAD", "\n# --- lever A alone, on the blind anchor ---\nRGB_SHARED_ENCODER: true       # one CNN trunk, N small action heads\nRGB_MONITOR_SENSITIVITY: true  # log train/rgb/pixel_sensitivity every update"
)
write(
    "cartpole_balance_nesy_rgb_shared_only.yaml",
    T1_ONLY_HDR
    % dict(
        env="CartpoleBalance RGB",
        base="cartpole_balance_nesy_rgb.yaml",
        anchor="results/rgb/ablation/cartpole/nesy_blind",
    ),
    strip_header(cb),
)

# walker blind anchor + shared. NOTE: walker_walk_nesy.yaml is the pure-STATE
# config -- it declares no RGB keys at all; rgb_pixel_ablation.py forces
# USE_RGB on, which is exactly how the committed walker/nesy_blind was produced.
wb = read("walker_walk_nesy.yaml")
wb = insert_after(
    wb,
    "META_POLICY_TYPE",
    "\n# --- lever A alone, on the blind anchor ---\n"
    "# walker_walk_nesy.yaml is the pure-state config: it declares no RGB keys and\n"
    "# rgb_pixel_ablation.py forces USE_RGB: True (this is how the committed\n"
    "# walker/nesy_blind arm was produced). RGB_SHARED_ENCODER is validated only\n"
    "# when USE_RGB is on, so it must be set alongside it here.\n"
    "USE_RGB: true\n"
    "RGB_SHARED_ENCODER: true       # one CNN trunk, N small action heads\n"
    "RGB_MONITOR_SENSITIVITY: true  # log train/rgb/pixel_sensitivity every update",
)
write(
    "walker_walk_nesy_shared_only.yaml",
    T1_ONLY_HDR
    % dict(
        env="WalkerWalk",
        base="walker_walk_nesy.yaml",
        anchor="results/rgb/ablation/walker/nesy_blind",
    ),
    strip_header(wb),
)

# ---------------- TIER 2: cartpole mirror of the metaz arm -------------------
T2_HDR = """# CartpoleBalance RGB: LEVER A + LEVER B, and NO auxiliary crutch.
#
# The cartpole mirror of walker_walk_nesy_rgb_shared_metaz_noaux.yaml. It is
# cartpole_balance_nesy_rgb_shared_noaux.yaml plus RGB_META_SEES_PIXELS: true,
# and nothing else, so the one-flag comparison against that file isolates lever B.
#
#   lever A  RGB_SHARED_ENCODER    one CNN trunk shared by all skill actors
#   lever B  RGB_META_SEES_PIXELS  meta-Q input = concatenate([state, latent])
#
# Lever B switches on two effects at once, and only one is new:
#   * the meta DECISION improves, because [state, latent] is strictly more
#     information than [state] alone;
#   * the meta TD error becomes a SECOND training signal on the shared encoder,
#     driven by the ENVIRONMENT reward rather than the hand-written skill rewards.
# To separate them, re-run this file with RGB_META_LATENT_STOP_GRAD: true, which
# keeps the improved decision and removes the gradient.
#
# THE STATE IS KEPT, NOT REPLACED: the hand-written symbolic precondition mask
# still reads the same privileged state, so the NeSy interpretability story is
# untouched -- this ADDS to the input of the boss rather than replacing it.
#
# WHAT TO WATCH
#   train/rgb/pixel_sensitivity    ~0 = still blind; must climb well above 0.01.
#   train/meta_encoder_grad_norm   a FLAT ZERO means the second signal is not
#                                  arriving and this arm answers a different
#                                  question than intended.
#
# NOTE FOR THE ABLATION: rgb_pixel_ablation.py feeds the meta the INTACT frames
# while corrupting only the actor stack, so its six conditions keep measuring
# ACTOR blindness and stay comparable with the pre-lever-B runs.
#
# INTEGRITY: train/rgb/aux_state_loss must be EXACTLY 0 for the whole run.
"""
t2c = read("cartpole_balance_nesy_rgb_shared_noaux.yaml")
t2c = insert_after(
    t2c,
    "RGB_SHARED_ENCODER",
    "RGB_META_SEES_PIXELS: true     # LEVER B: meta-Q input = [state, latent]",
)
write("cartpole_balance_nesy_rgb_shared_metaz_noaux.yaml", T2_HDR, strip_header(t2c))
