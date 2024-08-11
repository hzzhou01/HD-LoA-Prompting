# Roles Across Multiple Arguments (RAMS)

## Data

The data is split into train/dev/test files. Each line in a data file contains a json string. Each json contains:

* `ent_spans`: Start and end (inclusive) indices and an event/argument/role string.
* `evt_triggers`: Start and end (inclusive) indices and an event type string.
* `sentences`: Document text
* `gold_evt_links`: A triple of (event, argument, role) following the above format
* `source_url`: source of the text
* `split`: Which data split it belongs to
* `doc_key`: Which individual file it corresponds to (`nw_` is prepended on all of them)

All other fields are extraneous to allow for future iterations of RAMS

The `individual_files/{train, dev, test}` directory contains a human-readable file for each document. 

## Scorer

A scorer is released alongside the data.

The basic use of the scorer is below:

```
python scorer.py --gold_file <GOLD> --pred_file <PRED> --ontology_file <ONTOLOGY> --do_all
```

Some notes:

* <PRED> can be in one of two formats. In both cases, it contains one json string per line, and that json blob must contain a `doc_key`.
  1. It contains a `gold_evt_links` key, like in the gold data. Add the `--reuse_gold_format` flag when running the scorer.
  2. It contains a `predictions`, as in this example:
  ``
  "predictions": [[[70, 70], [63, 63, "victim", 1.0], [58, 58, "place", 1.0]]]
  ``
  It is a list of event-predictions (in RAMS there is only one). Each event-prediction starts with a [start, end] (inclusive) span for the event at index 0, and a [start, end, label, confidence] at subsequent indices for each argument.

* <ONTOLOGY> is used for type constrained decoding. a tsv where the 0th column is the event name, and the (2*i + 1)th column is the role name and (2*i + 2)th column is the count that is permitted by the event.
* Use `-cd` for type constrained decoding. Otherwise it is not on by default. 
* `--do_all` prints out metrics (`--metrics`), metrics by distance (`distance`), metrics by role (`role_table`), and csv confusion matrix (`confusion`). Individual metrics can be printed with their own flags (in parents).
* This is compatible with both Python2.7 and Python3.6

## Changelog

1.0: Re-shuffled splits to remove overlap, added two missing event types.
0.9: Initial Release