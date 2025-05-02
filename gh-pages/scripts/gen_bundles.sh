#!/usr/bin/env bash
TYPES=(votes heat flow cluster)
MODES=(norm cum)

for GAME in ./static/* ; do
	GAME="${GAME##*/}" # slug
	# where the SVGs live
	SRC="static/${GAME}"

	for typ in "${TYPES[@]}"; do
		for mode in "${MODES[@]}"; do
			# find all episodes for this type/mode
			pattern="${typ}_${mode}_*.svg"
			files=( $SRC/$pattern )
			for file in "${files[@]}"; do
				test ! -f "${file}" && continue
				base=$(basename "$file" .svg)            # e.g. cluster_cum_01
				IFS=_ read -r _t _m round <<<"$base"     # parse
				# create bundle dir
				DIR="content/verraeter/${GAME}/${typ}/${mode}/${round}"
				mkdir -p "$DIR"
				# write index.md if missing
				cat > "$DIR/index.md" <<EOF
---
title: "${typ^} - Runde ${round} $( [ "$mode" = cum ] && echo '(kumuliert)' )"
slug: "${round}"
type: "plot"
game: "${GAME}"
plottype: "${typ}"
mode: "${mode:-norm}"
round: "${round}"
maxRounds: ${#files[@]}
---
EOF
			done
		done
	done

	nr="${GAME#s}"
	nr="${nr#0}"
	cat > "content/verraeter/${GAME}/_index.md" <<EOF
---
title: "Die Verräter Staffel ${nr}"
slug: "${GAME}"
type: "overview"
---

Wähle einen Plottyp, Modus und Runde:
EOF
done
cat > "content/verraeter/_index.md" <<EOF
---
title: "Die Verräter"
---
EOF
