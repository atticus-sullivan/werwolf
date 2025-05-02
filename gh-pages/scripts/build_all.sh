#!/bin/bash

source ../venv/bin/activate

for dat in "$@" ; do
	mkdir -p "static/$(basename -s .yaml "${dat}")"

	python3 ../main.py check "$1"

	python3 ../main.py -o "static/$(basename -s .yaml "${dat}")/heat_norm_{}.svg" heatmap "$dat"

	python3 ../main.py -o "static/$(basename -s .yaml "${dat}")/flow_norm_{}.svg" flow "$dat"
	python3 ../main.py -o "static/$(basename -s .yaml "${dat}")/flow_cum_{}.svg" flow -c "$dat"

	python3 ../main.py -o "static/$(basename -s .yaml "${dat}")/cluster_norm_{}.svg" cluster "$dat"
	python3 ../main.py -o "static/$(basename -s .yaml "${dat}")/cluster_cum_{}.svg" cluster -c "$dat"

	python3 ../main.py -o "static/$(basename -s .yaml "${dat}")/votes_norm_{}.svg" votes "$dat"
	python3 ../main.py -o "static/$(basename -s .yaml "${dat}")/votes_cum_{}.svg" votes -c "$dat"
done
