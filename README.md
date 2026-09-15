# UniRule

Code for **Toward Unified Detection Rule Generation**.

## Code availability

This repository includes scripts for rule generation,
pairwise evaluation, Bradley–Terry scoring, and cross-setting analysis.

Detailed code will be made available upon publication of the paper.

## Scripts

| Script | Purpose |
| --- | --- |
| `generate_rule.py` | Generate a detection rule using semantic retrieval |
| `evaluate_rules.py` | Compare two candidate rules against the same requirement |
| `fit_bt.py` | Estimate Bradley-Terry method scores from pairwise judgments |
| `transfer_analysis.py` | Analyze prediction across languages and input forms |

## Usage

Requires Python 3.11 or later. Install dependencies with:

```bash
pip install -r requirements.txt
```

For example, score your own saved comparison records:

```bash
python fit_bt.py judgments.jsonl --output scores.json
```

Generation and evaluation require model services; generation also requires a
local embedding model and retrieval index.

## License

The source code in this repository is licensed under the [MIT License](LICENSE).

