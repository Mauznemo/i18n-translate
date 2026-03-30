#!/usr/bin/env python3
"""
i18n JSON AI Translator v1.5
Translates all text values in nested i18n JSON files using a local Ollama instance.

Usage:
  python3 i18n-translate.py <file> <ollama-url> --lang <language> --out <output-file> [options]
  python3 i18n-translate.py <file> <ollama-url> --config <config-file> [options]

Examples:
  python3 i18n-translate.py en.json http://localhost:11434 --lang German --out de.json
  python3 i18n-translate.py en.json http://localhost:11434 --lang Spanish --out es.json --model qwen3:8b
  python3 i18n-translate.py en.json http://localhost:11434 --lang French --out fr.json --scope pages.home
  python3 i18n-translate.py en.json http://localhost:11434 --lang German  --out de.json --missing-only
  python3 i18n-translate.py en.json http://localhost:11434 --lang German  --out de.json --no-think
  python3 i18n-translate.py en.json http://localhost:11434 --config languages.yaml
  python3 i18n-translate.py en.json http://localhost:11434 --config languages.json --missing-only
"""

import sys
import json
import re
import os
import urllib.request
import urllib.parse
import urllib.error
from typing import Optional

VERSION = "1.5"

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

# ── Config file loading ────────────────────────────────────────────────────────

def load_config(config_path: str) -> "list[dict]":
    """
    Load a YAML or JSON config file describing multiple target languages.

    Expected format (YAML):
      languages:
        - lang: German
          out: de.json
        - lang: "German (Du Form)"
          out: de-informal.json
        - lang: Spanish
          out: es.json
          scope: pages.home   # optional per-language scope override

    Expected format (JSON):
      {
        "languages": [
          { "lang": "German", "out": "de.json" },
          { "lang": "Spanish", "out": "es.json", "scope": "pages.home" }
        ]
      }

    Returns a list of dicts, each with at minimum "lang" and "out" keys.
    Optional keys per entry: "scope".
    """
    if not os.path.isfile(config_path):
        print(f"{RED}Config file not found:{RESET} {config_path}")
        sys.exit(1)

    ext = os.path.splitext(config_path)[1].lower()

    # ── JSON config ────────────────────────────────────────────────────────────
    if ext == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                raw = json.load(f)
            except json.JSONDecodeError as e:
                print(f"{RED}Invalid JSON in config file:{RESET} {e}")
                sys.exit(1)
        entries = raw.get("languages") if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            print(f"{RED}Config JSON must have a top-level 'languages' array.{RESET}")
            sys.exit(1)
        return _validate_config_entries(entries, config_path)

    # ── YAML config (stdlib-only, no PyYAML required) ──────────────────────────
    if ext in (".yaml", ".yml"):
        return _load_yaml_config(config_path)

    print(f"{RED}Unsupported config format '{ext}'. Use .json, .yaml, or .yml{RESET}")
    sys.exit(1)


def _validate_config_entries(entries: list, source: str) -> "list[dict]":
    valid = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            print(f"{YELLOW}Warning: config entry #{i + 1} is not an object — skipping.{RESET}")
            continue
        if "lang" not in entry or "out" not in entry:
            print(f"{YELLOW}Warning: config entry #{i + 1} missing 'lang' or 'out' — skipping.{RESET}")
            continue
        valid.append(entry)
    if not valid:
        print(f"{RED}No valid language entries found in config file: {source}{RESET}")
        sys.exit(1)
    return valid


def _load_yaml_config(config_path: str) -> "list[dict]":
    """
    Minimal YAML parser sufficient for the config format — no external deps.
    Supports only the simple flat structure this tool needs:

      languages:
        - lang: German
          out: de.json
        - lang: Spanish
          out: es.json
          scope: pages.home

    Quoted strings (single or double) are supported.
    Inline comments (#) are stripped.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entries = []
    current: Optional[dict] = None
    in_languages = False

    for raw_line in lines:
        # Strip inline comments and trailing whitespace
        line = re.sub(r'\s+#.*$', '', raw_line.rstrip())
        stripped = line.lstrip()

        if not stripped:
            continue

        # Detect "languages:" section header
        if re.match(r'^languages\s*:', line):
            in_languages = True
            continue

        if not in_languages:
            continue

        # New list item
        if re.match(r'^\s+-\s+', line):
            if current is not None:
                entries.append(current)
            current = {}
            # The first key may appear on the same line as the dash
            rest = re.sub(r'^\s+-\s+', '', line)
            _yaml_parse_kv(rest, current)
            continue

        # Continuation key on its own line (indented, no dash)
        if current is not None and re.match(r'^\s+\S', line):
            _yaml_parse_kv(stripped, current)
            continue

        # Top-level key outside a list item — ignore (e.g. another section)

    if current is not None:
        entries.append(current)

    return _validate_config_entries(entries, config_path)


def _yaml_parse_kv(text: str, target: dict) -> None:
    """Parse a single 'key: value' line into target dict."""
    m = re.match(r'^([\w-]+)\s*:\s*(.*)', text.strip())
    if not m:
        return
    key = m.group(1)
    value = m.group(2).strip()
    # Strip surrounding quotes
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    target[key] = value

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
    for key, source_value in source_chunk:
        translated_value = translated_map.get(key)
        if not translated_value or not isinstance(translated_value, str):
            continue
        if re.search(r'\{[\w:]+\}|%[sd]|%\(\w+\)s|<[a-zA-Z/]', source_value):
            continue
        word_count = len(source_value.split())
        if word_count > GLOSSARY_MAX_WORDS:
            continue
        src = source_value.strip()
        tgt = translated_value.strip()
        if src and tgt and src != tgt:
            if src not in glossary and len(glossary) >= GLOSSARY_MAX_ENTRIES:
                oldest = next(iter(glossary))
                del glossary[oldest]
            glossary[src] = tgt


def seed_glossary_from_existing(glossary: dict,
                                source_data,
                                existing_data) -> int:
    source_entries = flatten_json(source_data)
    seeded = 0
    for key, source_value in source_entries:
        if is_missing_or_empty(existing_data, key):
            continue
        translated_value = get_nested_value(existing_data, key)
        if not translated_value or not isinstance(translated_value, str):
            continue
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
    system = build_system_prompt(target_language, no_think, glossary)
    payload = {k: v for k, v in entries}
    user_prompt = (
        f"Translate the following JSON values to {target_language}. "
        f"Return ONLY valid JSON with the same keys and translated values.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    raw = ollama_chat(base_url, model, system, user_prompt, no_think)
    raw = re.sub(r'^```[a-z]*\n?', '', raw.strip())
    raw = re.sub(r'\n?```$', '', raw.strip())
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
    system = build_system_prompt(target_language, no_think, glossary)
    result = {}
    for key, value in entries:
        user_prompt = (
            f"Translate this UI string to {target_language}. "
            f"Output ONLY the translated string, nothing else.\n\n{value}"
        )
        translated = ollama_chat(base_url, model, system, user_prompt, no_think)
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
    print(f"  python3 i18n-translate.py <file> <ollama-url> --config <config-file> [options]")
    print()
    print(f"{BOLD}Required arguments:{RESET}")
    print(f"  file               Path to your source i18n JSON file (e.g. en.json)")
    print(f"  ollama-url         URL of your Ollama server  (e.g. http://localhost:11434)")
    print()
    print(f"{BOLD}Single-language mode (mutually exclusive with --config):{RESET}")
    print(f"  --lang <language>  Target language name       (e.g. German, Spanish, French)")
    print(f"  --out  <file>      Output file path           (e.g. de.json)")
    print()
    print(f"{BOLD}Batch mode:{RESET}")
    print(f"  --config <file>    Path to a YAML or JSON config file listing all target languages.")
    print(f"                     See 'Config file format' below.")
    print()
    print(f"{BOLD}Optional (apply to both modes):{RESET}")
    print(f"  --model <name>     Ollama model to use (e.g. qwen3:8b). If omitted you will be prompted.")
    print(f"  --scope <key>      Only translate strings under this key (e.g. pages or pages.home).")
    print(f"                     Per-language scope in the config overrides this.")
    print(f"  --chunk-size <n>   Strings per API call (default: {CHUNK_SIZE}). Lower = more accurate.")
    print(f"  --missing-only     Only translate keys absent or empty in the output file.")
    print(f"  --no-think         Disable chain-of-thought reasoning (faster; for models like qwen3,")
    print(f"                     deepseek-r1). Passes think:false to Ollama and prepends /no_think")
    print(f"                     to the system prompt. Safe to use with non-reasoning models.")
    print()
    print(f"{BOLD}Config file format:{RESET}")
    print(f"  YAML (languages.yaml):")
    print(f"    languages:")
    print(f"      - lang: German")
    print(f"        out: de.json")
    print(f'      - lang: "German (Du Form)"')
    print(f"        out: de-informal.json")
    print(f"      - lang: Spanish")
    print(f"        out: es.json")
    print(f"        scope: pages.home   # optional, overrides --scope for this language")
    print()
    print(f"  JSON (languages.json):")
    print(f'    {{"languages": [')
    print(f'      {{"lang": "German",  "out": "de.json"}},')
    print(f'      {{"lang": "Spanish", "out": "es.json", "scope": "pages.home"}}')
    print(f"    ]}}")
    print()
    print(f"{BOLD}Examples:{RESET}")
    print(f"  python3 i18n-translate.py en.json http://localhost:11434 --lang German --out de.json")
    print(f"  python3 i18n-translate.py en.json http://localhost:11434 --lang Spanish --out es.json --model qwen3:8b")
    print(f"  python3 i18n-translate.py en.json http://localhost:11434 --lang French  --out fr.json --scope pages.home")
    print(f"  python3 i18n-translate.py en.json http://localhost:11434 --lang German  --out de.json --missing-only --no-think")
    print(f"  python3 i18n-translate.py en.json http://localhost:11434 --config languages.yaml")
    print(f"  python3 i18n-translate.py en.json http://localhost:11434 --config languages.yaml --missing-only --model qwen3:8b")

def progress_bar(done: int, total: int, width: int = 36) -> str:
    pct   = done / total if total else 0
    filled = int(width * pct)
    bar   = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {done}/{total} ({pct:.0%})"

# ── Core translation runner ────────────────────────────────────────────────────

def run_translation(
    filepath: str,
    base_url: str,
    lang: str,
    out_file: str,
    model: str,
    scope: Optional[str],
    chunk_size: int,
    missing_only: bool,
    no_think: bool,
    source_data: dict,
    label: Optional[str] = None,
) -> bool:
    """
    Translate source_data into `lang` and write result to `out_file`.
    Returns True on success, False if nothing was translated.
    """
    # ── Build entry list ───────────────────────────────────────────────────────
    if scope:
        try:
            scoped_data = get_scoped_data(source_data, scope)
        except KeyError as e:
            print(f"  {RED}Error:{RESET} {e}")
            return False
        entries = flatten_json(scoped_data, prefix=scope)
    else:
        entries = flatten_json(source_data)

    total_source = len(entries)
    if total_source == 0:
        print(f"  {YELLOW}No translatable strings found.{RESET}")
        return False

    # ── Build output data ──────────────────────────────────────────────────────
    if os.path.isfile(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            try:
                out_data = json.load(f)
            except json.JSONDecodeError:
                print(f"  {YELLOW}Warning: existing output file is invalid JSON — starting fresh.{RESET}")
                out_data = deep_copy_structure(source_data)
    else:
        out_data = deep_copy_structure(source_data)

    # ── Filter missing keys ────────────────────────────────────────────────────
    if missing_only:
        entries_before = len(entries)
        entries = [(k, v) for k, v in entries if is_missing_or_empty(out_data, k)]
        skipped_existing = entries_before - len(entries)
        print(f"  Found {BOLD}{total_source}{RESET} source strings.")
        if skipped_existing:
            print(f"  {DIM}Skipping {skipped_existing} already-translated string(s).{RESET}")
        if not entries:
            print(f"  {GREEN}{BOLD}Nothing to translate — all keys already present in {out_file}{RESET}")
            return True
        print(f"  {YELLOW}{BOLD}{len(entries)} missing/empty string(s) to translate.{RESET}\n")
    else:
        print(f"  Found {BOLD}{total_source}{RESET} translatable strings.\n")

    # ── Translate in chunks ────────────────────────────────────────────────────
    errors   = 0
    done     = 0
    total    = len(entries)
    chunks   = [entries[i:i + chunk_size] for i in range(0, total, chunk_size)]

    glossary: dict = {}
    seeded = seed_glossary_from_existing(glossary, source_data, out_data)
    if seeded:
        print(f"  {DIM}Glossary pre-seeded with {seeded} term(s) from existing translations.{RESET}\n")

    print(f"  Translating with {BOLD}{model}{RESET}...\n")

    for chunk_idx, chunk in enumerate(chunks):
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
                short_key = key.split(".")[-1]
                translated_value = translated_map.get(short_key)
            if translated_value and isinstance(translated_value, str):
                set_nested_value(out_data, key, translated_value)
            else:
                errors += 1
            done += 1

        update_glossary(glossary, chunk, translated_map)
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
    return True

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
    config     = None
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
        elif a == "--config" and i + 1 < len(args):
            config = args[i + 1]; i += 2
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

    # ── Validate mode ──────────────────────────────────────────────────────────
    if config and (lang or out_file):
        print(f"{RED}--config cannot be combined with --lang or --out.{RESET}")
        print(f"{DIM}Use --scope, --missing-only, --no-think, --model, --chunk-size with --config.{RESET}")
        sys.exit(1)

    if not config:
        missing = []
        if not lang:     missing.append("--lang")
        if not out_file: missing.append("--out")
        if missing:
            print(f"{RED}Missing required argument(s):{RESET} {', '.join(missing)}")
            print(f"{DIM}Tip: use --config <file> to translate multiple languages at once.{RESET}")
            print_usage()
            sys.exit(1)

    if not os.path.isfile(filepath):
        print(f"{RED}File not found:{RESET} {filepath}")
        sys.exit(1)

    print_header()

    # ── Load source JSON ───────────────────────────────────────────────────────
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            source_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"{RED}Invalid JSON:{RESET} {e}")
            sys.exit(1)

    # ── Pick model ─────────────────────────────────────────────────────────────
    models = ollama_list_models(base_url)
    if not models:
        print(f"{RED}No models found on Ollama server at {base_url}{RESET}")
        sys.exit(1)

    if model:
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

    # ══════════════════════════════════════════════════════════════════════════
    # BATCH MODE
    # ══════════════════════════════════════════════════════════════════════════
    if config:
        language_entries = load_config(config)
        total_langs = len(language_entries)

        print(f"  {DIM}Source:{RESET}    {filepath}")
        print(f"  {DIM}Config:{RESET}    {config}")
        print(f"  {DIM}Model:{RESET}     {model}")
        print(f"  {DIM}Server:{RESET}    {base_url}")
        print(f"  {DIM}Chunk size:{RESET} {chunk_size} strings/call")
        if scope:
            print(f"  {DIM}Scope:{RESET}     {BOLD}{scope}{RESET} {DIM}(global default){RESET}")
        if missing_only:
            print(f"  {DIM}Mode:{RESET}      {YELLOW}Missing keys only{RESET}")
        if no_think:
            print(f"  {DIM}Reasoning:{RESET} {YELLOW}Disabled (--no-think){RESET}")
        print(f"\n  {BOLD}Batch:{RESET} {total_langs} language(s) to process\n")
        print(f"  {'─' * 62}")

        success_count = 0
        for idx, entry in enumerate(language_entries, 1):
            entry_lang  = entry["lang"]
            entry_out   = entry["out"]
            # Per-language scope overrides the global --scope
            entry_scope = entry.get("scope") or scope

            print(f"\n  {BOLD}{CYAN}[{idx}/{total_langs}] {entry_lang}{RESET}  →  {entry_out}")
            if entry_scope:
                print(f"  {DIM}Scope: {entry_scope}{RESET}")
            print()

            ok = run_translation(
                filepath=filepath,
                base_url=base_url,
                lang=entry_lang,
                out_file=entry_out,
                model=model,
                scope=entry_scope,
                chunk_size=chunk_size,
                missing_only=missing_only,
                no_think=no_think,
                source_data=source_data,
                label=f"{idx}/{total_langs}",
            )
            if ok:
                success_count += 1

        # ── Batch summary ──────────────────────────────────────────────────────
        print(f"  {'═' * 62}")
        print(f"  {BOLD}Batch complete:{RESET} {GREEN}{success_count}/{total_langs} language(s) processed{RESET}\n")
        sys.exit(0)

    # ══════════════════════════════════════════════════════════════════════════
    # SINGLE-LANGUAGE MODE
    # ══════════════════════════════════════════════════════════════════════════
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

    run_translation(
        filepath=filepath,
        base_url=base_url,
        lang=lang,
        out_file=out_file,
        model=model,
        scope=scope,
        chunk_size=chunk_size,
        missing_only=missing_only,
        no_think=no_think,
        source_data=source_data,
    )


if __name__ == "__main__":
    main()