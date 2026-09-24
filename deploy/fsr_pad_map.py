"""FSR mux CSV → sim tactile channel → USD pad link (Trial 15 training asset).

Trial 15 / PadTac+BT trained on::

  shadow_padtac_biotac.usd
  = shadow_padtac.usd pads + BioTac distal tips

Sim channel assignment is NOT in the USD file.  Contact on body
``rh_fsr_pad_CXX`` is scattered into the 24-d tactile vector by
``PAD_LINK_TO_CHANNEL`` / ``PAD_BT_LINK_TO_CHANNEL`` in
``roto/tasks/robots/shadowlite/shadowlite.py``.

Those channels were authored together with the USD in
``scripts/make_padtac_usd.py`` (pad name, parent link, sim_channel).

Hardware mux index = Arduino CSV column order (C0..C11).  The USD has
**no mux indices**.  Wire-check must confirm: pressing the physical pad
at the USD parent site lights the mux row whose ``sim_ch`` equals
``PAD_LINK_TO_CHANNEL[usd_link]``.

Three mux wires are NOT equal to the USD Cxx suffix (corrected 15 Jul)::

  mux C5  → USD rh_fsr_pad_C11 (rfmid / rh_rfmiddle, ch 13)
  mux C6  → USD rh_fsr_pad_C05 (palm, ch 2)
  mux C11 → USD rh_fsr_pad_C06 (ffmid / rh_ffmiddle, ch 11)
"""

from __future__ import print_function

# From scripts/make_padtac_usd.py PADS (what was written into shadow_padtac.usd).
# Trial 15 uses the same 12 pads inside shadow_padtac_biotac.usd.
# mux: Arduino CSV column (HW). usd/parent/sim_ch: sim training contract.
FSR_PAD_ENTRIES = [
    # mux, site_name, sim_ch, usd_suffix, parent_link_in_USD
    {"mux": 0, "name": "thprox", "sim_ch": 10, "usd": "C00", "parent": "rh_thproximal"},
    {"mux": 1, "name": "ffprox", "sim_ch": 7, "usd": "C01", "parent": "rh_ffproximal"},
    {"mux": 2, "name": "mfknuckle", "sim_ch": 4, "usd": "C02", "parent": "rh_palm"},
    {"mux": 3, "name": "rfprox", "sim_ch": 9, "usd": "C03", "parent": "rh_rfproximal"},
    {"mux": 4, "name": "rfknuckle", "sim_ch": 5, "usd": "C04", "parent": "rh_palm"},
    {"mux": 5, "name": "rfmid", "sim_ch": 13, "usd": "C11", "parent": "rh_rfmiddle"},
    {"mux": 6, "name": "palm", "sim_ch": 2, "usd": "C05", "parent": "rh_palm"},
    {"mux": 7, "name": "ffknuckle", "sim_ch": 3, "usd": "C07", "parent": "rh_palm"},
    {"mux": 8, "name": "mfprox", "sim_ch": 8, "usd": "C08", "parent": "rh_mfproximal"},
    {"mux": 9, "name": "thmiddle", "sim_ch": 18, "usd": "C09", "parent": "rh_thmiddle"},
    {"mux": 10, "name": "mfmid", "sim_ch": 12, "usd": "C10", "parent": "rh_mfmiddle"},
    {"mux": 11, "name": "ffmid", "sim_ch": 11, "usd": "C06", "parent": "rh_ffmiddle"},
]

N_FSR = len(FSR_PAD_ENTRIES)

# Deploy: FSR_CHANNELS[mux_i] = sim tactile channel for that CSV column
FSR_CHANNELS = [e["sim_ch"] for e in FSR_PAD_ENTRIES]

FSR_NAMES = [
    "C%d_%s" % (e["mux"], e["name"]) for e in FSR_PAD_ENTRIES
]

# Visualizer: (mux_label, site_name, sim_ch, usd_link)
PAD_META = [
    ("C%d" % e["mux"], e["name"], e["sim_ch"], "rh_fsr_pad_%s" % e["usd"])
    for e in FSR_PAD_ENTRIES
]

# USD link -> sim ch (must equal shadowlite.PAD_LINK_TO_CHANNEL)
EXPECTED_USD_CHANNELS = {
    "rh_fsr_pad_%s" % e["usd"]: e["sim_ch"] for e in FSR_PAD_ENTRIES
}


def print_mapping_table():
    """Print mux / site / sim ch / USD link / parent (wire-check reference)."""
    print("Trial 15 asset: shadow_padtac_biotac.usd (pads from shadow_padtac.usd)")
    print("Sim: contact on USD pad link -> PAD_LINK_TO_CHANNEL -> 24-d tactile")
    print("HW:  Arduino mux CSV column -> FSR_CHANNELS[mux] -> same 24-d index")
    print("-" * 88)
    print(
        "%-6s %-12s %6s  %-18s %s"
        % ("mux", "site", "sim_ch", "USD pad link", "USD parent")
    )
    print("-" * 88)
    for e in FSR_PAD_ENTRIES:
        print(
            "C%-5d %-12s %6d  rh_fsr_pad_%-6s %s"
            % (e["mux"], e["name"], e["sim_ch"], e["usd"], e["parent"])
        )
    print("-" * 88)
    print("Wire-check: press ONE physical pad at the USD parent site;")
    print("  only that mux row should spike, and its sim_ch must match training.")
    print("Mux C5/C6/C11 are NOT USD C05/C06/C11 (HW wire remap).")


def verify_against_shadowlite():
    """Return mismatch strings vs shadowlite.PAD_LINK_TO_CHANNEL."""
    try:
        from roto.tasks.robots.shadowlite.shadowlite import PAD_LINK_TO_CHANNEL
    except ImportError:
        return ["(skip: roto / torch not importable in this Python)"]

    errs = []
    for link, ch in EXPECTED_USD_CHANNELS.items():
        if PAD_LINK_TO_CHANNEL.get(link) != ch:
            errs.append(
                "%s: map has ch %d, shadowlite has %s"
                % (link, ch, PAD_LINK_TO_CHANNEL.get(link))
            )
    for link, ch in PAD_LINK_TO_CHANNEL.items():
        if link.startswith("rh_fsr_pad_") and link not in EXPECTED_USD_CHANNELS:
            errs.append("shadowlite has extra pad link %s" % link)
    return errs


def verify_against_make_padtac():
    """Return mismatches vs scripts/make_padtac_usd.py PADS (USD authoring)."""
    # Inline copy of make_padtac_usd.PADS (pad, parent, xyz, channel)
    authored = {
        "rh_fsr_pad_C00": (10, "rh_thproximal"),
        "rh_fsr_pad_C01": (7, "rh_ffproximal"),
        "rh_fsr_pad_C02": (4, "rh_palm"),
        "rh_fsr_pad_C03": (9, "rh_rfproximal"),
        "rh_fsr_pad_C04": (5, "rh_palm"),
        "rh_fsr_pad_C05": (2, "rh_palm"),
        "rh_fsr_pad_C06": (11, "rh_ffmiddle"),
        "rh_fsr_pad_C07": (3, "rh_palm"),
        "rh_fsr_pad_C08": (8, "rh_mfproximal"),
        "rh_fsr_pad_C09": (18, "rh_thmiddle"),
        "rh_fsr_pad_C10": (12, "rh_mfmiddle"),
        "rh_fsr_pad_C11": (13, "rh_rfmiddle"),
    }
    errs = []
    for e in FSR_PAD_ENTRIES:
        link = "rh_fsr_pad_%s" % e["usd"]
        if link not in authored:
            errs.append("unknown USD link %s" % link)
            continue
        ch, parent = authored[link]
        if ch != e["sim_ch"]:
            errs.append("%s: map ch %d != authored ch %d" % (link, e["sim_ch"], ch))
        if parent != e["parent"]:
            errs.append("%s: map parent %s != authored %s" % (link, e["parent"], parent))
    if len(EXPECTED_USD_CHANNELS) != len(authored):
        errs.append("pad count mismatch map vs make_padtac_usd")
    return errs


if __name__ == "__main__":
    print_mapping_table()
    print("\nCheck vs make_padtac_usd.py (USD authoring):")
    a = verify_against_make_padtac()
    print("  PASS" if not a else "\n".join("  FAIL: " + x for x in a))
    print("\nCheck vs shadowlite.PAD_LINK_TO_CHANNEL (training scatter):")
    b = verify_against_shadowlite()
    if b and b[0].startswith("(skip"):
        print("  ", b[0])
    else:
        print("  PASS" if not b else "\n".join("  FAIL: " + x for x in b))
