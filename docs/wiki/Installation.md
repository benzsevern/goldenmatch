# Installation

## Basic Install

```bash
pip install goldenmatch
```

This installs the core tool with file-based deduplication and matching.

## Optional Extras

GoldenMatch has optional dependency groups for additional features:

```bash
# Embedding-based semantic matching (sentence-transformers + FAISS)
pip install goldenmatch[embeddings]

# LLM-powered accuracy boost (Claude/GPT-4 for pair labeling)
pip install goldenmatch[llm]

# Postgres database integration
pip install goldenmatch[postgres]

# Everything
pip install goldenmatch[embeddings,llm,postgres]
```

## Development Install

```bash
git clone https://github.com/benzsevern/goldenmatch.git
cd goldenmatch
pip install -e ".[dev,embeddings,llm,postgres]"
```

## Requirements

- Python 3.11+
- Core dependencies: Polars, RapidFuzz, Typer, Textual, PyYAML, Pydantic, jellyfish, numpy, openpyxl

## Verify Installation

```bash
goldenmatch --help
```
