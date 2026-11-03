"""Ego4D FHO verb taxonomy (the closed action-verb schema of the hand-action task) and an alias/inflection normalizer.

The 118 verb classes of the Ego4D Forecasting Hands & Objects benchmark (https://ego4d-data.org/docs/benchmarks/hands-and-objects/).
"""

from __future__ import annotations

from functools import cache

EGO4D_FHO_VERBS: tuple[str, ...] = (
    "adjust_(regulate,_increase/reduce,_change)",
    "apply_(spread,_smear)",
    "arrange_(straighten,_sort,_distribute,_align)",
    "attach_(plug-in,_join,_fasten,_connect,_attach)",
    "blow",
    "break",
    "carry",
    "catch",
    "clap",
    "clean_(sweep,_scrub,_mop,_dust)",
    "climb",
    "close",
    "consume_(taste,_sip,_eat,_drink)",
    "count",
    "cover",
    "cross",
    "crochet",
    "cut_(trim,_slice,_chop)",
    "detach_(unplug,_unhook,_disconnect)",
    "dig",
    "dip",
    "divide_(split,_separate)",
    "draw",
    "drill",
    "drive_(ride,_drive)",
    "enter",
    "feed",
    "file_(with_tool)",
    "fill",
    "fold",
    "fry",
    "give",
    "grate",
    "grind",
    "hang",
    "hit_(knock,_hit,_hammer)",
    "hold_(support,_grip,_grasp)",
    "insert",
    "inspect_(check,_look,_examine,_view)",
    "iron",
    "kick",
    "knead",
    "knit",
    "lift",
    "lock",
    "loosen",
    "mark",
    "measure_(weigh,_measure)",
    "mix",
    "mold",
    "move_(transfer,_pass,_exchange)",
    "open",
    "operate_(use,_dial,_click-button)",
    "pack",
    "paint",
    "park",
    "peel",
    "pet",
    "plant",
    "play",
    "point",
    "pour",
    "press",
    "pull",
    "pump",
    "push",
    "put_(place,_leave,_drop)",
    "read",
    "remove",
    "repair",
    "roll",
    "sand",
    "scoop",
    "scrape",
    "screw",
    "scroll",
    "search",
    "serve",
    "sew_(weave,_stitch,_sew)",
    "shake",
    "sharpen",
    "shuffle",
    "sieve",
    "sit",
    "smooth",
    "spray",
    "sprinkle",
    "squeeze",
    "stand",
    "step",
    "stick_(tape,_stick,_glue)",
    "stretch",
    "swing",
    "take_(pick,_grab,_get)",
    "talk_(talk,_interact,_converse)",
    "throw_(toss,_dump,_dispose)",
    "tie",
    "tighten",
    "tilt",
    "touch",
    "turn_(spin,_rotate,_flip,_turn_over)",
    "turn_off_(turn_off,_switch_off)",
    "turn_on_(switch_on,_start,_light)",
    "uncover",
    "unfold",
    "unroll",
    "unscrew",
    "untie",
    "walk",
    "wash",
    "water",
    "wear",
    "weld",
    "wipe",
    "write",
    "zip",
    "watch",
    "wave",
)

EGO4D_FHO_VERB_SET = frozenset(EGO4D_FHO_VERBS)


def normalize_ego4d_fho_verb(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text in EGO4D_FHO_VERB_SET:
        return text
    alias_lookup = _ego4d_fho_verb_alias_lookup()
    for key in _simple_verb_key_variants(text):
        verb = alias_lookup.get(key)
        if verb is not None:
            return verb
    return None


def ego4d_fho_verb_alias_keys(verb: str) -> set[str]:
    if verb not in EGO4D_FHO_VERB_SET:
        return set()
    return _ego4d_fho_verb_alias_keys(verb)


def _ego4d_fho_verb_alias_keys(verb: str) -> set[str]:
    keys = _simple_verb_key_variants(verb)
    head, separator, tail = verb.partition("_(")
    keys.update(_simple_verb_key_variants(head))
    if separator:
        aliases = tail.removesuffix(")")
        for alias_group in aliases.split(","):
            for alias in alias_group.split("/"):
                keys.update(_simple_verb_key_variants(alias))
    return {variant for key in keys for variant in _regular_inflection_keys(key)}


@cache
def _ego4d_fho_verb_alias_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for verb in EGO4D_FHO_VERBS:
        for key in _ego4d_fho_verb_alias_keys(verb):
            lookup.setdefault(key, verb)
    return lookup


def _regular_inflection_keys(key: str) -> set[str]:
    variants = {key}
    if not key:
        return variants

    if key.endswith("ie"):
        variants.add(f"{key[:-2]}ying")
    elif key.endswith("e") and not key.endswith("ee"):
        variants.add(f"{key[:-1]}ing")
    else:
        variants.add(f"{key}ing")

    if key.endswith(("s", "x", "z", "ch", "sh")):
        variants.add(f"{key}es")
    elif len(key) > 1 and key.endswith("y") and key[-2] not in "aeiou":
        variants.add(f"{key[:-1]}ies")
    else:
        variants.add(f"{key}s")

    if key.endswith("e"):
        variants.add(f"{key}d")
    elif len(key) > 1 and key.endswith("y") and key[-2] not in "aeiou":
        variants.add(f"{key[:-1]}ied")
    else:
        variants.add(f"{key}ed")
    return variants


def _simple_verb_key_variants(value: str) -> set[str]:
    key = _simple_verb_key(value)
    if not key:
        return set()
    variants = {key}
    if len(key) > 4 and key.endswith("ing"):
        stem = key[:-3]
        variants.add(stem)
        variants.add(f"{stem}e")
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            variants.add(stem[:-1])
    if len(key) > 3 and key.endswith("ed"):
        stem = key[:-2]
        variants.add(stem)
        variants.add(f"{stem}e")
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            variants.add(stem[:-1])
    if len(key) > 3 and key.endswith("s"):
        variants.add(key[:-1])
    return {variant for variant in variants if variant}


def _simple_verb_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())
