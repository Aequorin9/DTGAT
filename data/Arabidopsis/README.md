# Arabidopsis input data

This directory contains the processed Arabidopsis input files used by the
DTGAT example.

## `expression_matrix.csv`

- Rows represent Arabidopsis genes.
- The first column, `gene_id`, contains Arabidopsis gene identifiers.
- The remaining columns represent expression samples identified by their run
  accessions.
- The training script reads `gene_id` as the row index and standardizes each
  expression feature before model training.

## `reference_network.csv`

The file contains reference TF-target regulatory relationships with the
following columns:

- `TF_ID`: transcription factor identifier;
- `Target_ID`: target-gene identifier;
- `type`: source/category label retained from the processed reference network.

Only relationships for which both genes occur in the expression matrix are
retained by the training script.

## Data handling in the script

The positive regulatory relationships are split into training, validation,
and test sets at a ratio of 60%/20%/20% using the predefined seed. Negative
examples are sampled from TF-target pairs absent from the complete positive
reference network. Only training positive edges are used to construct the
graph supplied to the model. Validation relationships are used for checkpoint
selection, and test relationships are reserved for final evaluation.

The original data sources and detailed preprocessing procedures are described
in the associated manuscript.

