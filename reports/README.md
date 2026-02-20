# Reports

Output and reports from RAG_MED runs (QA results, ValueAI evaluation summaries).

## Typical contents

- **qa_result.json** — Generated question–answer pairs (can be saved here via `rag-med generate ... -o reports/qa_result.json`).
- **\*_valueai_eval.json** — ValueAI RAG evaluation summaries when using `--valueai-eval`.

## Note

This directory can be kept in version control or added to `.gitignore` if you prefer not to commit generated outputs. Add a `.gitkeep` to keep the folder in git when empty.
