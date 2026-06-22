# Cultural Heritage Perception Analysis

Cultural Heritage Perception Analysis provides LLM-based pipelines for
quantifying how tourists perceive cultural heritage assets in review text and
images. The package supports two complementary tasks:

1. Extracting and scoring cultural value evidence from tourist reviews across
   eight dimensions: social, economic, political, historic, aesthetical,
   scientific, age, and ecological.
2. Assessing the visual quality of heritage attraction images from the
   perspective of a prospective tourist.

## Repository Structure

```text
.
├── src/heritage_perception_analysis/
│   ├── dimensions.py
│   ├── text_pipeline.py
│   ├── image_pipeline.py
│   ├── llm_client.py
│   ├── memory.py
│   └── cli.py
├── examples/
│   ├── reviews.csv
│   ├── test.jpg
│   └── image_memory/
│       ├── 1.jpg
│       ├── 2.jpg
│       └── 3.jpg
├── tests/
├── data/
└── outputs/
```

## Installation

```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuration

The package uses an OpenAI-compatible chat completions API. Set credentials and
model names with environment variables:

```
export LLM_API_KEY="your_api_key"
export LLM_BASE_URL="https://your-compatible-endpoint/v1"
export LLM_MODEL="your_text_model"
export VISION_API_KEY="your_vision_api_key_if_different"
export VISION_BASE_URL="https://your-vision-compatible-endpoint/v1"
export VISION_MODEL="your_vision_model"
```

Set `LLM_BASE_URL` and `VISION_BASE_URL` to the API root, not the full
`/chat/completions` endpoint. For example, GLM/BigModel's OpenAI-compatible
root is:

```
export VISION_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
export VISION_MODEL="glm-4.6v-flash"
```

`VISION_API_KEY` and `VISION_BASE_URL` are optional when the same provider
supports both text and image models. If the image command returns an error such
as `unknown variant image_url`, the configured vision endpoint or model is
text-only; set `VISION_BASE_URL` and `VISION_MODEL` to a vision-capable
OpenAI-compatible chat model.

`LLM_BASE_URL` and `VISION_BASE_URL` are optional for the default OpenAI
endpoint. If your provider supports JSON mode, you can also set:

```
export LLM_JSON_MODE=true
```

## Quick Smoke Tests

Run commands from the repository root. The `main.py` commands call the
configured model API and require the environment variables above.

```
python main.py --help
python main.py text --input examples/reviews.csv --output outputs/review_scores.csv --text-column comments --limit 1 --no-resume
python main.py images --image-root examples --output outputs/image_quality_scores.csv --limit-per-folder 1 --no-resume
```

Add `--print-model-io` to either analysis command to inspect the exact messages
sent to the model and the raw model response.

## Text Review Analysis

Input CSV files must contain a review text column. The default column name is
`comments`.

```
python main.py text --input examples/reviews.csv --output outputs/review_scores.csv --text-column comments
```

The text pipeline has three stages:

1. Input and contextual calibration: the three
   most recent successful analyses.
2. Preliminary extraction: candidate dimension-wise evidence copied from the
   source review.
3. Verification and scoring: hallucination filtering, relevance validation, and
   scoring from 1 to 10. Missing verified evidence is written as `N/A` status
   with a blank score.

Output columns include `identified_dimensions`, `verified_json`, one score
column per dimension, one verified evidence column per dimension, status
fields, processing time, and token usage.

To inspect the exact model input and output for each text stage, add
`--print-model-io`:

```
python main.py text --input examples/reviews.csv --output outputs/review_scores.csv --limit 1 --print-model-io
```

## Image Visual Quality Analysis

Place images either directly under a root folder or inside first-level
attraction folders. The repository separates image examples into memory seeds
and test inputs. `examples/image_memory/1.jpg`, `examples/image_memory/2.jpg`,
and `examples/image_memory/3.jpg` initialize the image pipeline's rolling
contextual memory; later successful analyses replace them in the memory window.
The image discovery step skips `image_memory` so those calibration images are
not scored as test inputs. `examples/test.jpg` is a sample input for a quick
smoke test.

```
python main.py images --image-root examples --output outputs/image_quality_scores.csv --limit-per-folder 1
```

For multiple test images, remove `--limit-per-folder 1` or set a larger value.
Each successful image analysis is appended to the rolling contextual memory, and
the next image uses the most recent three successful examples.

The image pipeline has three stages:

1. Data input: local image files are encoded as base64 data URIs.
2. Contextual memory: the three most recent successful image analyses are used
   as calibration examples.
3. Visual quality assessment: the model returns `visual_analysis`, `score`, and
   `reason` as JSON.

To inspect the model input and output for image analysis, add
`--print-model-io`. Base64 image data is summarized by default to keep terminal
output readable:

```
python main.py images --image-root examples --output outputs/image_quality_scores.csv --limit-per-folder 1 --print-model-io
```

Use `--include-image-data-uri` together with `--print-model-io` only when the
full base64 image payload is needed.

## Data and Credentials

API keys are read from environment variables and are not stored in source files.
Generated files should be written under `outputs/` or another local output
folder. The default `.gitignore` excludes local data, generated outputs, cache
files, virtual environments, and `.env` files.

Input data can be stored under `data/` for local runs. The tracked `.gitkeep`
files preserve the folder structure without publishing local datasets.

## Development

Run unit tests:

```
PYTHONPATH=src python -m unittest discover -s tests
```

The unit tests do not call external model APIs and do not require API keys.

Run syntax checks:

```
python -m compileall src tests
```
