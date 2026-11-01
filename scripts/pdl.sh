#!/bin/bash
# pdl.sh URL OUT [N=16] [CHUNK_MB=64]
# Parallel ranged download: N workers fetch CHUNK_MB pieces straight into OUT.
# - A chunk counts only if the server answers HTTP 206 with the exact byte count
#   (Google Drive sometimes answers a range with a 2 KB "Quota exceeded" HTML page).
# - Resumable: rerun the same command; finished chunks are kept ($OUT.chunks/*.ok).
# Retries per chunk: PDL_TRIES (default 60), longest wait between tries PDL_WAIT s (default 20).
# Google Drive: use https://drive.usercontent.google.com/download?id=FILE_ID&export=download&confirm=t
#   A popular file hits Drive's "Quota exceeded" limit (every request gets the HTML page);
#   wait it out with e.g. PDL_TRIES=100000 PDL_WAIT=300 and N=4, or download signed in (rclone).
set -u
# Stopping the script must stop its xargs workers too: run as its own process group (setsid)
# and kill the whole group on exit or signal.
if [ "$(ps -o pgid= $$ | tr -d ' ')" != "$$" ]; then exec setsid "$0" "$@"; fi
trap 'trap - TERM EXIT; kill -- -$$ 2>/dev/null' INT TERM EXIT
TRIES=${PDL_TRIES:-60}; WAIT=${PDL_WAIT:-20}
U=$1; OUT=$2; N=${3:-16}; CH=$(( ${4:-64} * 1048576 ))
D=$OUT.chunks
[ -f "$OUT" ] && [ ! -d "$D" ] && { echo "$OUT already complete"; exit 0; }
SIZE=$(curl -sL -r 0-0 -D - -o /dev/null "$U" | grep -i '^content-range' | tail -1 | sed 's#.*/##' | tr -dc 0-9)
[ "${SIZE:-0}" -gt 0 ] 2>/dev/null || SIZE=$(curl -sIL "$U" | grep -i '^content-length' | tail -1 | tr -dc 0-9)
[ "${SIZE:-0}" -gt 0 ] 2>/dev/null || { echo "cannot get file size (no range support?)"; exit 1; }
NC=$(( (SIZE + CH - 1) / CH )); mkdir -p "$D"; touch "$OUT"; t0=$(date +%s)
[ "$(stat -c %s "$OUT")" -eq "$SIZE" ] || truncate -s "$SIZE" "$OUT"
echo "$OUT: $SIZE bytes, $NC chunks of $(( CH / 1048576 )) MB, $N workers"

chunk() {
  local i=$1 s=$(( $1 * CH )) e=$(( ($1 + 1) * CH - 1 )) code try
  [ $e -ge $SIZE ] && e=$((SIZE - 1))
  local want=$((e - s + 1)) t=$D/c$i.tmp ok=$D/c$i.ok
  [ -f "$ok" ] && return 0
  for try in $(seq 1 $TRIES); do
    code=$(curl -sL --max-time 1800 --speed-limit 10000 --speed-time 60 -r $s-$e -o "$t" -w '%{http_code}' "$U")
    if [ "$code" = 206 ] && [ "$(stat -c %s "$t" 2>/dev/null || echo 0)" -eq "$want" ]; then
      dd if="$t" of="$OUT" bs=4M seek=$s oflag=seek_bytes conv=notrunc status=none \
        && rm -f "$t" && touch "$ok" && return 0
    fi
    sleep $(( try * 2 < WAIT ? try * 2 : WAIT ))
  done
  echo "chunk $i failed after $TRIES tries"; return 1
}
export -f chunk; export U OUT SIZE CH D TRIES WAIT

( while sleep 30; do
    k=$(ls "$D" 2>/dev/null | grep -c '\.ok$'); el=$(( $(date +%s) - t0 ))
    echo "$(date +%T) $k/$NC chunks, $(( k * CH / 1048576 / (el + 1) )) MB/s avg"
  done ) & MON=$!
seq 0 $((NC - 1)) | xargs -P "$N" -I{} bash -c 'chunk {}'
kill $MON 2>/dev/null

k=$(ls "$D" | grep -c '\.ok$')
[ "$k" -eq "$NC" ] || { echo "INCOMPLETE: $k/$NC chunks; rerun the same command to resume"; exit 1; }
rm -r "$D"; echo "done $OUT $SIZE bytes in $(( $(date +%s) - t0 )) s"
