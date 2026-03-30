#!/usr/bin/env python3
"""
i18n JSON AI Translator v1.4
Translates all text values in nested i18n JSON files using a local Ollama instance.

Usage:
  python3 i18n-translate.py <file> <ollama-url> --lang <language> --out <output-file> [options]

Examples:
  python3 i18n-translate.py en.json http://localhost:11434 --lang German --out de.json
  python3 i18n-translate.py en.json http://localhost:11434 --lang Spanish --out es.json --model qwen3:8b
  python3 i18n-translate.py en.json http://localhost:11434 --lang French --out fr.json --scope pages.home
  python3 i18n-translate.py en.json http://localhost:11434 --lang German  --out de.json --missing-only
  python3 i18n-translate.py en.json http://localhost:11434 --lang German  --out de.json --no-think
"""

import sys
import json
import re
import os
import urllib.request
import urllib.parse
import urllib.error
from typing import Optional

VERSION = "1.4"

# ANSI colors
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# ── Ollama API ─────────────────────────────────────────────────────────────────

def ollama_list_models(base_url: str) -> "list[str]":
    req = urllib.request.Request(f"{base_url}/api/tags")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]
    except urllib.error.URLError as e:
        print(f"\n{RED}Error connecting to Ollama: {e}{RESET}")
        print(f"{DIM}Make sure Ollama is running at {base_url}{RESET}")
        sys.exit(1)

def ollama_chat(base_url: str, model: str, system: str, user: str, no_think: bool = False) -> str:
    payload = json.dumps({
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "options": {
            "temperature": 0.1,  # Low temp for consistent, accurate translation
        },
        # Some Ollama backends honour a top-level "think" flag directly.
        # Setting it to False is a no-op on models that don't support it.
        **({"think": False} if no_think else {}),
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["message"]["content"].strip()
    except urllib.error.URLError as e:
        print(f"\n{RED}Error calling Ollama: {e}{RESET}")
        sys.exit(1)

# ── JSON helpers ───────────────────────────────────────────────────────────────

def flatten_json(data, prefix="") -> list:
    items = []
    if isinstance(data, dict):
        for k, v in data.items():
            full_key = f"{prefix}.{k}" if prefix else k
            items.extend(flatten_json(v, full_key))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            items.extend(flatten_json(v, f"{prefix}[{i}]"))
    elif isinstance(data, str) and data.strip():
        items.append((prefix, data))
    return items

def get_scoped_data(data: dict, scope: str):
    parts = scope.split(".")
    obj = data
    for part in parts:
        if not isinstance(obj, dict) or part not in obj:
            raise KeyError(f"Scope key '{scope}' not found in JSON (failed at '{part}')")
        obj = obj[part]
    return obj

def set_nested_value(data, key_path: str, new_value: str):
    parts = re.split(r'\.(?![^\[]*\])', key_path)
    obj = data
    for part in parts[:-1]:
        m = re.match(r'^(.*)\[(\d+)\]$', part)
        if m:
            key, idx = m.group(1), int(m.group(2))
            if key not in obj:
                obj[key] = []
            while len(obj[key]) <= idx:
                obj[key].append({})
            obj = obj[key][idx]
        else:
            if part not in obj or not isinstance(obj[part], dict):
                obj[part] = {}
            obj = obj[part]
    last = parts[-1]
    m = re.match(r'^(.*)\[(\d+)\]$', last)
    if m:
        key, idx = m.group(1), int(m.group(2))
        if key not in obj:
            obj[key] = []
        while len(obj[key]) <= idx:
            obj[key].append("")
        obj[key][idx] = new_value
    else:
        obj[last] = new_value

def deep_copy_structure(data):
    """Deep copy JSON-serialisable structure."""
    return json.loads(json.dumps(data))

def get_nested_value(data, key_path: str):
    """
    Walk into data using the same dotted/bracketed key_path used by set_nested_value.
    Returns the value, or None if any part of the path is missing.
    """
    parts = re.split(r'\.(?![^\[]*\])', key_path)
    obj = data
    for part in parts:
        m = re.match(r'^(.*)\[(\d+)\]$', part)
        if m:
            key, idx = m.group(1), int(m.group(2))
            if not isinstance(obj, dict) or key not in obj:
                return None
            obj = obj[key]
            if not isinstance(obj, list) or idx >= len(obj):
                return None
            obj = obj[idx]
        else:
            if not isinstance(obj, dict) or part not in obj:
                return None
            obj = obj[part]
    return obj

def is_missing_or_empty(data, key_path: str) -> bool:
    """Return True if the key doesn't exist in data or its value is an empty/whitespace string."""
    val = get_nested_value(data, key_path)
    if val is None:
        return True
    if isinstance(val, str) and not val.strip():
        return True
    return False

# ── Translation strategy ───────────────────────────────────────────────────────

# The /no_think prefix is the standard Ollama convention for disabling chain-of-thought
# reasoning on models that support it (e.g. qwen3, deepseek-r1). It is prepended to the
# system prompt when --no-think is passed. On models that don't support it the prefix is
# harmless and is simply ignored.
NO_THINK_PREFIX = "/no_think\n"

SYSTEM_PROMPT = """\
You are a professional i18n (internationalisation) translator. Your job is to translate UI strings from a software application.

Rules you MUST follow:
1. Preserve ALL placeholders exactly as-is. Placeholders look like: {name}, {count}, $date, {{escaped}}, %s, %d, %(key)s.
2. Preserve ALL HTML tags exactly as-is (e.g. <strong>, <br/>, <a href="...">, etc.).
3. Preserve ALL special characters, punctuation style, and capitalisation conventions of the original.
4. Keep the same tone: if the original is formal, stay formal; if casual, stay casual.
5. Never add explanations, notes, or commentary — output ONLY the translated text.
6. Never translate key names, only values.
7. If given a JSON object, respond ONLY with valid JSON — no markdown fences, no extra text.
8. Translate to {target_language}.
"""

def build_system_prompt(target_language: str, no_think: bool = False,
                        glossary: Optional[dict] = None) -> str:
    prompt = SYSTEM_PROMPT.replace("{target_language}", target_language)
    if glossary:
        lines = "\n".join(f"  {src} → {tgt}" for src, tgt in glossary.items())
        prompt += (
            f"\nConsistency glossary — you MUST use these established translations exactly:\n"
            f"{lines}\n"
        )
    if no_think:
        prompt = NO_THINK_PREFIX + prompt
    return prompt


# Chunk size: number of key-value pairs per API call.
# Smaller = more accurate but slower. Larger = faster but may lose context.
CHUNK_SIZE = 20

# Glossary: only track short UI terms (≤ this many words) to keep the prompt tight.
GLOSSARY_MAX_WORDS = 5
# Cap total glossary entries injected into each prompt.
GLOSSARY_MAX_ENTRIES = 50


def update_glossary(glossary: dict,
                    source_chunk: "list[tuple[str, str]]",
                    translated_map: dict) -> None:
    """
    After a chunk is translated, add short source→translation pairs to the
    running glossary so future chunks stay consistent.

    Only short values (≤ GLOSSARY_MAX_WORDS words, no placeholders, no HTML)
    are added — these are the UI terms most prone to inconsistent rendering
    across chunks (e.g. "Submit", "Cancel", "Save changes").

    The glossary is capped at GLOSSARY_MAX_ENTRIES; oldest entries are evicted
    first when the cap is reached.
    """
    for key, source_value in source_chunk:
        translated_value = translated_map.get(key)
        if not translated_value or not isinstance(translated_value, str):
            continue
        # Skip strings with placeholders or HTML tags — they're context-specific.
        if re.search(r'\{[\w:]+\}|%[sd]|%\(\w+\)s|<[a-zA-Z/]', source_value):
            continue
        word_count = len(source_value.split())
        if word_count > GLOSSARY_MAX_WORDS:
            continue
        src = source_value.strip()
        tgt = translated_value.strip()
        if src and tgt and src != tgt:
            # Evict oldest entry if at cap (dicts preserve insertion order in Python 3.7+)
            if src not in glossary and len(glossary) >= GLOSSARY_MAX_ENTRIES:
                oldest = next(iter(glossary))
                del glossary[oldest]
            glossary[src] = tgt


def seed_glossary_from_existing(glossary: dict,
                                source_data,
                                existing_data) -> int:
    """
    Pre-populate the glossary from translations that already exist in the
    output file.  For every string in source_data that has a corresponding
    non-empty translation in existing_data, we treat the pair as a confirmed
    source→translation term and add it to the glossary (subject to the same
    word-count and placeholder filters used by update_glossary).

    Returns the number of entries seeded.
    """
    source_entries = flatten_json(source_data)
    seeded = 0
    for key, source_value in source_entries:
        if is_missing_or_empty(existing_data, key):
            continue
        translated_value = get_nested_value(existing_data, key)
        if not translated_value or not isinstance(translated_value, str):
            continue
        # Apply the same filters as update_glossary
        if re.search(r'\{[\w:]+\}|%[sd]|%\(\w+\)s|<[a-zA-Z/]', source_value):
            continue
        if len(source_value.split()) > GLOSSARY_MAX_WORDS:
            continue
        src = source_value.strip()
        tgt = translated_value.strip()
        if src and tgt and src != tgt:
            if src not in glossary and len(glossary) >= GLOSSARY_MAX_ENTRIES:
                oldest = next(iter(glossary))
                del glossary[oldest]
            glossary[src] = tgt
            seeded += 1
    return seeded


def translate_chunk(entries: "list[tuple[str, str]]", target_language: str,
                    base_url: str, model: str, no_think: bool = False,
                    glossary: Optional[dict] = None) -> dict:
    """
    Translate a list of (key, value) pairs in one API call.
    Returns a dict mapping key -> translated value.
    """
    system = build_system_prompt(target_language, no_think, glossary)

    # Build a compact JSON object: { key: value, ... }
    payload = {k: v for k, v in entries}
    user_prompt = (
        f"Translate the following JSON values to {target_language}. "
        f"Return ONLY valid JSON with the same keys and translated values.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    raw = ollama_chat(base_url, model, system, user_prompt, no_think)

    # Strip accidental markdown fences
    raw = re.sub(r'^```[a-z]*\n?', '', raw.strip())
    raw = re.sub(r'\n?```$', '', raw.strip())

    # Strip any residual <think>…</think> block that leaked into the content
    raw = re.sub(r'<think>.*?</think>\s*', '', raw, flags=re.DOTALL).strip()

    try:
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("Response is not a JSON object")
        return result
    except (json.JSONDecodeError, ValueError) as e:
        print(f"\n{YELLOW}Warning: Could not parse chunk response as JSON ({e}). "
              f"Falling back to string-by-string mode for this chunk.{RESET}")
        return translate_chunk_fallback(entries, target_language, base_url, model, no_think, glossary)


def translate_chunk_fallback(entries: "list[tuple[str, str]]", target_language: str,
                              base_url: str, model: str, no_think: bool = False,
                              glossary: Optional[dict] = None) -> dict:
    """Translate each string individually as a fallback."""
    system = build_system_prompt(target_language, no_think, glossary)
    result = {}
    for key, value in entries:
        user_prompt = (
            f"Translate this UI string to {target_language}. "
            f"Output ONLY the translated string, nothing else.\n\n{value}"
        )
        translated = ollama_chat(base_url, model, system, user_prompt, no_think)
        # Strip any residual <think>…</think> block that leaked into the content
        translated = re.sub(r'<think>.*?</think>\s*', '', translated, flags=re.DOTALL).strip()
        result[key] = translated
    return result

# ── Interactive model picker ───────────────────────────────────────────────────

def pick_model(models: "list[str]") -> str:
    print(f"\n  {BOLD}Available Ollama models:{RESET}\n")
    for i, m in enumerate(models, 1):
        print(f"    {CYAN}({i}){RESET} {m}")
    print()
    while True:
        try:
            raw = input(f"  Select model (1-{len(models)}): ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(0)
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            return models[int(raw) - 1]
        print(f"  {RED}Invalid choice.{RESET} Enter a number between 1 and {len(models)}.")

# ── Display ────────────────────────────────────────────────────────────────────

def print_header():
    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║        i18n AI Translator                ║{RESET}")
    print(f"{BOLD}{CYAN}║                        v{VERSION:<17}║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════╝{RESET}\n")

def print_usage():
    print(f"{BOLD}Usage:{RESET}")
    print(f"  python3 i18n-translate.py <file> <ollama-url> --lang <language> --out <output-file> [options]")
    print()
    print(f"{BOLD}Required arguments:{RESET}")
    print(f"  file               Path to your source i18n JSON file (e.g. en.json)")
    print(f"  ollama-url         URL of your Ollama server  (e.g. http://localhost:11434)")
    print(f"  --lang <language>  Target language name       (e.g. German, Spanish, French)")
    print(f"  --out  <file>      Output file path           (e.g. de.json)")
    print()
    print(f"{BOLD}Optional:{RESET}")
    print(f"  --model <name>     Ollama model to use (e.g. qwen3:8b). If omitted you will be prompted.")
    print(f"  --scope <key>      Only translate strings under this key (e.g. pages or pages.home)")
    print(f"  --chunk-size <n>   Strings per API call (default: {CHUNK_SIZE}). Lower = more accurate.")
    print(f"  --missing-only     Only translate keys absent or empty in the output file.")
    print(f"  --no-think         Disable chain-of-thought reasoning (faster; for models like qwen3,")
    print(f"                     deepseek-r1). Passes think:false to Ollama and prepends /no_think")
    print(f"                     to the system prompt. Safe to use with non-reasoning models.")
    print()
    print(f"{BOLD}Examples:{RESET}")
    print(f"  python3 i18n-translate.py en.json http://localhost:11434 --lang German --out de.json")
    print(f"  python3 i18n-translate.py en.json http://localhost:11434 --lang Spanish --out es.json --model qwen3:8b")
    print(f"  python3 i18n-translate.py en.json http://localhost:11434 --lang French  --out fr.json --scope pages.home")
    print(f"  python3 i18n-translate.py en.json http://localhost:11434 --lang German  --out de.json --missing-only")
    print(f"  python3 i18n-translate.py en.json http://localhost:11434 --lang German  --out de.json --model qwen3:8b --no-think")

def progress_bar(done: int, total: int, width: int = 36) -> str:
    pct   = done / total if total else 0
    filled = int(width * pct)
    bar   = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {done}/{total} ({pct:.0%})"

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print_header()
        print_usage()
        sys.exit(0 if args else 1)

    if len(args) < 2:
        print_usage()
        sys.exit(1)

    filepath   = args[0]
    base_url   = args[1].rstrip("/")
    lang       = None
    out_file   = None
    model      = None
    scope      = None
    chunk_size = CHUNK_SIZE
    missing_only = False
    no_think     = False

    i = 2
    while i < len(args):
        a = args[i]
        if a == "--lang" and i + 1 < len(args):
            lang = args[i + 1]; i += 2
        elif a == "--out" and i + 1 < len(args):
            out_file = args[i + 1]; i += 2
        elif a == "--model" and i + 1 < len(args):
            model = args[i + 1]; i += 2
        elif a == "--scope" and i + 1 < len(args):
            scope = args[i + 1]; i += 2
        elif a == "--missing-only":
            missing_only = True; i += 1
        elif a == "--no-think":
            no_think = True; i += 1
        elif a == "--chunk-size" and i + 1 < len(args):
            try:
                chunk_size = int(args[i + 1])
            except ValueError:
                print(f"{RED}--chunk-size must be an integer{RESET}")
                sys.exit(1)
            i += 2
        else:
            print(f"{RED}Unknown argument:{RESET} {a}")
            print_usage()
            sys.exit(1)

    # Validate required args
    missing = []
    if not lang:     missing.append("--lang")
    if not out_file: missing.append("--out")
    if missing:
        print(f"{RED}Missing required argument(s):{RESET} {', '.join(missing)}")
        print_usage()
        sys.exit(1)

    if not os.path.isfile(filepath):
        print(f"{RED}File not found:{RESET} {filepath}")
        sys.exit(1)

    print_header()

    # ── Load JSON ──────────────────────────────────────────────────────────────
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"{RED}Invalid JSON:{RESET} {e}")
            sys.exit(1)

    # ── Pick model ─────────────────────────────────────────────────────────────
    models = ollama_list_models(base_url)
    if not models:
        print(f"{RED}No models found on Ollama server at {base_url}{RESET}")
        sys.exit(1)

    if model:
        # Validate the passed model exists (fuzzy: prefix match)
        matched = next((m for m in models if m == model or m.startswith(model)), None)
        if not matched:
            print(f"{YELLOW}Warning: model '{model}' not found. Available:{RESET}")
            for m in models:
                print(f"  - {m}")
            print()
            model = pick_model(models)
        else:
            model = matched
    else:
        model = pick_model(models)

    # ── Print config ───────────────────────────────────────────────────────────
    print(f"\n  {DIM}Source:{RESET}    {filepath}")
    print(f"  {DIM}Output:{RESET}    {out_file}")
    print(f"  {DIM}Language:{RESET}  {lang}")
    print(f"  {DIM}Model:{RESET}     {model}")
    print(f"  {DIM}Server:{RESET}    {base_url}")
    print(f"  {DIM}Chunk size:{RESET} {chunk_size} strings/call")
    if scope:
        print(f"  {DIM}Scope:{RESET}     {BOLD}{scope}{RESET}")
    if missing_only:
        print(f"  {DIM}Mode:{RESET}      {YELLOW}Missing keys only{RESET}")
    if no_think:
        print(f"  {DIM}Reasoning:{RESET} {YELLOW}Disabled (--no-think){RESET}")
    print()

    # ── Build entry list ───────────────────────────────────────────────────────
    if scope:
        try:
            scoped_data = get_scoped_data(data, scope)
        except KeyError as e:
            print(f"{RED}Error:{RESET} {e}")
            sys.exit(1)
        entries = flatten_json(scoped_data, prefix=scope)
    else:
        entries = flatten_json(data)

    total_source = len(entries)
    if total_source == 0:
        print(f"{YELLOW}No translatable strings found.{RESET}")
        sys.exit(0)

    # ── Build output data (start as deep copy of source) ──────────────────────
    # If the output file already exists, load it so we can merge into it.
    if os.path.isfile(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            try:
                out_data = json.load(f)
            except json.JSONDecodeError:
                print(f"{YELLOW}Warning: existing output file is invalid JSON — starting fresh.{RESET}")
                out_data = deep_copy_structure(data)
    else:
        out_data = deep_copy_structure(data)

    # ── Filter to missing/empty keys if requested ──────────────────────────────
    if missing_only:
        entries_before = len(entries)
        entries = [(k, v) for k, v in entries if is_missing_or_empty(out_data, k)]
        skipped_existing = entries_before - len(entries)
        print(f"  Found {BOLD}{total_source}{RESET} source strings.")
        if skipped_existing:
            print(f"  {DIM}Skipping {skipped_existing} already-translated string(s).{RESET}")
        if not entries:
            print(f"\n  {GREEN}{BOLD}Nothing to translate — all keys already present in {out_file}{RESET}\n")
            sys.exit(0)
        print(f"  {YELLOW}{BOLD}{len(entries)} missing/empty string(s) to translate.{RESET}\n")
    else:
        print(f"  Found {BOLD}{total_source}{RESET} translatable strings.\n")

    # ── Translate in chunks ────────────────────────────────────────────────────
    errors   = 0
    done     = 0
    total    = len(entries)
    chunks   = [entries[i:i + chunk_size] for i in range(0, total, chunk_size)]

    # Running glossary: pre-seeded from any translations already present in the
    # output file so --missing-only runs stay consistent with prior work,
    # then grown chunk-by-chunk during this run.
    glossary = {}  # type: dict
    seeded = seed_glossary_from_existing(glossary, data, out_data)
    if seeded:
        print(f"  {DIM}Glossary pre-seeded with {seeded} term(s) from existing translations.{RESET}\n")

    print(f"  Translating with {BOLD}{model}{RESET}...\n")

    for chunk_idx, chunk in enumerate(chunks):
        # Show progress
        print(f"\r\033[K  {progress_bar(done, total)}  chunk {chunk_idx + 1}/{len(chunks)}", end="", flush=True)

        try:
            translated_map = translate_chunk(chunk, lang, base_url, model, no_think,
                                             glossary if glossary else None)
        except SystemExit:
            raise
        except Exception as e:
            print(f"\n  {RED}Chunk {chunk_idx + 1} failed: {e}{RESET}")
            errors += 1
            done += len(chunk)
            continue

        for key, original_value in chunk:
            translated_value = translated_map.get(key)

            if translated_value is None:
                # Key might be present without scope prefix in response
                short_key = key.split(".")[-1]
                translated_value = translated_map.get(short_key)

            if translated_value and isinstance(translated_value, str):
                set_nested_value(out_data, key, translated_value)
            else:
                # Keep original if translation missing
                errors += 1

            done += 1

        # Update the glossary with short terms from this chunk so the next
        # chunk can reuse them for consistency.
        update_glossary(glossary, chunk, translated_map)

        # Refresh progress after chunk
        print(f"\r\033[K  {progress_bar(done, total)}  chunk {chunk_idx + 1}/{len(chunks)}", end="", flush=True)

    print(f"\r\033[K  {progress_bar(total, total)}  Done!                        ")

    # ── Save output ────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # ── Summary ────────────────────────────────────────────────────────────────
    translated_count = total - errors
    print(f"\n  {'═' * 62}")
    print(f"  {BOLD}Summary{RESET}")
    print(
        f"  {GREEN}Translated:{RESET} {translated_count}  "
        f"{RED}Errors/skipped:{RESET} {errors}  "
        f"{DIM}Total: {total}{RESET}"
    )
    if glossary:
        total_terms = len(glossary)
        grown = total_terms - seeded
        if seeded and grown:
            print(f"  {DIM}Glossary: {seeded} pre-seeded + {grown} new = {total_terms} terms{RESET}")
    print(f"\n  {GREEN}{BOLD}Output saved to {out_file}{RESET}\n")


if __name__ == "__main__":
    main()