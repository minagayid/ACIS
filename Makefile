.PHONY: test compile validate evaluate

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

compile:
	python -m compileall -q src

validate:
	PYTHONPATH=src python -m acis.cli validate --recordings data/examples/recordings.jsonl --annotations data/examples/annotations.jsonl

evaluate:
	PYTHONPATH=src python -m acis.cli evaluate --recordings data/examples/recordings.jsonl --annotations data/examples/annotations.jsonl --output evaluations/example-report.md --allow-synthetic
