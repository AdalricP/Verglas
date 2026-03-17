# Verglas

Verglas is a very simple .musicXML sheet generator that was built on top of GPT-2 using the Leider Corpus converted to the musicXML.

## Data

The project uses the [Leider Corpus MXL](https://github.com/AdalricP/LeiderCorpusMXL) - a MusicXML conversion of the OpenScore Lieder Corpus for those who would find it useful.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python scripts/train.py
```

The trained model will be saved to `model/verglas.pt`.

### Generation

```bash
python scripts/generate.py
```

Saves to output.mxl

## License

MIT
