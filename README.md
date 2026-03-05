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

### Run it
(Replace the Ollama URL)
```sh
i18n-translate en.json http://localhost:11434 --lang German --out de.json
i18n-translate en.json http://localhost:11434 --lang Spanish --out es.json --model qwen3:8b
i18n-translate en.json http://localhost:11434 --lang French --out fr.json --scope pages.home
```
### or
```sh
python3 i18n-translate.py en.json http://localhost:11434 --lang German --out de.json
python3 i18n-translate.py en.json http://localhost:11434 --lang Spanish --out es.json --model qwen3:8b
python3 i18n-translate.py en.json http://localhost:11434 --lang French --out fr.json --scope pages.home
```