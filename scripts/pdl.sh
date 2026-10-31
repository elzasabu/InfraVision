#!/bin/bash
# pdl.sh URL OUT N : download with N parallel range requests, verify size.
set -u
U=$1; OUT=$2; N=${3:-16}
SIZE=$(curl -sIL "$U" | grep -i '^content-length' | tail -1 | tr -dc 0-9)
C=$(( (SIZE + N - 1) / N )); P=$OUT.parts; mkdir -p "$P"; t0=$(date +%s)
part() { local i=$1 s=$(( $1 * C )) e=$(( ($1 + 1) * C - 1 )); [ $e -ge $SIZE ] && e=$((SIZE - 1))
  local want=$((e - s + 1)) f=$P/p$(printf %03d $i)
  for try in 1 2 3 4 5 6; do
    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [ "$have" -eq "$want" ] && return 0
    [ "$have" -gt "$want" ] && have=0 && rm -f "$f"
    curl -sL --max-time 1800 -r $((s + have))-$e "$U" >> "$f"
  done
  [ "$(stat -c %s "$f")" -eq "$want" ] || { echo "part $i incomplete"; return 1; }; }
for i in $(seq 0 $((N - 1))); do part $i & done; wait
cat "$P"/p* > "$OUT"
[ "$(stat -c %s "$OUT")" -eq "$SIZE" ] || { echo "size mismatch"; exit 1; }
rm -r "$P"; echo "done $OUT $SIZE bytes in $(( $(date +%s) - t0 )) s"
