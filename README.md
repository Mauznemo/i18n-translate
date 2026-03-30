# i18n-translate
Simple terminal tool to translate your **i18n** JSON localization files using Ollama.

> [!NOTE]
> I only tested this on macOS.

## Installation
Just download the `i18n-translate.py` file to your computer and follow the next steps.

### Make it executable
```sh
chmod +x i18n-translate.py
```

### Optionally move it somewhere on your PATH
```sh
mv i18n-translate.py /usr/local/bin/i18n-translate
```

---

## Usage

### Single language
Translate into one language at a time:

```sh
i18n-translate en.json http://localhost:11434 --lang German --out de.json
i18n-translate en.json http://localhost:11434 --lang Spanish --out es.json --model qwen3:8b
i18n-translate en.json http://localhost:11434 --lang French --out fr.json --scope pages.home
i18n-translate en.json http://localhost:11434 --lang German --out de.json --missing-only --no-think
```

### Batch mode (multiple languages at once)
Point the tool at a config file and it translates every language in one run — great for updating all locales when you add a new key:

```sh
i18n-translate en.json http://localhost:11434 --config languages.yaml
i18n-translate en.json http://localhost:11434 --config languages.yaml --missing-only
i18n-translate en.json http://localhost:11434 --config languages.yaml --model qwen3:8b --no-think
```

All other flags (`--missing-only`, `--no-think`, `--model`, `--scope`, `--chunk-size`) work in batch mode and apply to every language. A `scope` defined inside the config file overrides the global `--scope` for that specific language.

---

## Config file format

### YAML (`languages.yaml`)
```yaml
languages:
  - lang: German
    out: de.json

  - lang: "German (Du Form)"   # quotes needed for special characters
    out: de-informal.json

  - lang: Spanish
    out: es.json

  - lang: French
    out: fr.json
    scope: pages.home           # optional: only translate this subtree for French
```

### JSON (`languages.json`)
```json
{
  "languages": [
    { "lang": "German",          "out": "de.json" },
    { "lang": "German (Du Form)", "out": "de-informal.json" },
    { "lang": "Spanish",         "out": "es.json" },
    { "lang": "French",          "out": "fr.json", "scope": "pages.home" }
  ]
}
```

> [!TIP]
> The config file requires no Ollama URL or model — those are still passed on the command line, so you can swap models without editing the file.

---

## All options

| Flag                | Description                                                                        |
| ------------------- | ---------------------------------------------------------------------------------- |
| `--lang <language>` | Target language name (single-language mode)                                        |
| `--out <file>`      | Output file path (single-language mode)                                            |
| `--config <file>`   | YAML or JSON config for batch mode (replaces `--lang` + `--out`)                   |
| `--model <name>`    | Ollama model to use (e.g. `qwen3:8b`). Prompted interactively if omitted.          |
| `--scope <key>`     | Only translate strings under this dotted key (e.g. `pages` or `pages.home`)        |
| `--chunk-size <n>`  | Strings per API call (default: 20). Lower = more accurate, slower.                 |
| `--missing-only`    | Skip keys that already have a translation in the output file                       |
| `--no-think`        | Disable chain-of-thought reasoning — faster for models like `qwen3`, `deepseek-r1` |

> [!NOTE]
> `--config` and `--lang`/`--out` are mutually exclusive.

---

## or with `python3`
```sh
python3 i18n-translate.py en.json http://localhost:11434 --lang German --out de.json
python3 i18n-translate.py en.json http://localhost:11434 --lang Spanish --out es.json --model qwen3:8b
python3 i18n-translate.py en.json http://localhost:11434 --lang French --out fr.json --scope pages.home
python3 i18n-translate.py en.json http://localhost:11434 --lang German --out de.json --missing-only --no-think
python3 i18n-translate.py en.json http://localhost:11434 --config languages.yaml
python3 i18n-translate.py en.json http://localhost:11434 --config languages.yaml --missing-only --model qwen3:8b
```