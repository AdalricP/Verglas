# Verglas

Verglas is a very simple .musicXML sheet generator that was built on top of GPT-2 using the Leider Corpus converted to the musicXML.

## Data

The project uses the [Leider Corpus MXL](https://github.com/AdalricP/LeiderCorpusMXL) - a MusicXML conversion of the OpenScore Lieder Corpus for those who would find it useful.

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python scripts/train.py
```

The trained model will be saved to `model/verglas.pt`.

## Generation

```bash
python scripts/generate.py
```

Output will be saved to `output.xml`.

## Project Structure

```
verglas/
├── data/           # Training data (.mxl files)
├── model/          # Saved models
├── scripts/        # Training and generation scripts
└── src/            # Source code
```

## License

MIT
