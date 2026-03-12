# AFSO GEPA Dataset

This folder contains a small GEPA-oriented dataset built from three AFSO program documents for the Anatole France school project.

Files:
- `afso_requirements_gepa_v1.jsonl`: 48 structured QA examples.

Design choices:
- `48` examples total.
- `16` examples per category:
  - `fonctionnel`
  - `technique_performance_env`
  - `maintenance`
- Split:
  - `30` train
  - `9` val
  - `9` test

Recommended task:
- Input: `question` + retrieved `context`
- Output: short `answer` plus the structured fields already present in each row

Suggested metric:
- `answer` exactness or strong semantic match
- `value` exact match
- `unit` exact match when applicable
- `theme` exact match
- `requirement_type` exact match
- `source_pages` overlap bonus

Source page codes:
- `Fxx`: functional program PDF pages
- `Txx`: technical / environmental / performance PDF pages
- `Mxx`: operation / maintenance PDF pages

Notes:
- Context snippets are concise evidence summaries, not full page extracts.
- The dataset is intentionally small and stable so GEPA can improve prompt behavior without requiring a large labeling effort.
