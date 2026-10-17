#!/usr/bin/env bash
# Supervisor: keep the Qwen3-VL download alive, then score GenEval2.
#
# Two things bit us already, so this handles both:
#   * hf-mirror's xet presigned URLs expire after an hour, killing long pulls with
#     AccessDenied. If no downloader is alive and the weights are incomplete, we
#     restart it -- snapshot_download resumes from the .incomplete files.
#   * readiness must be read from the files, not from `pgrep`: an earlier watcher
#     grepped for a string that appeared in its own command line and never fired.
set -u

MODEL=/physis/models/Qwen3-VL-8B-Instruct
ROOT=/physis/unified-dopsd/mmada-opsd-clean
PY=/physis/unified-dopsd/venv/bin/python
LOG=/physis/unified-dopsd/scoring.log
DL=/tmp/restart_dl.py

log () { echo "$(date -u '+%F %T UTC'): $*" >> "$LOG"; }

weights_ready () {
  [ -f "$MODEL/model.safetensors.index.json" ] || return 1
  [ "$(find "$MODEL/.cache/huggingface/download" -name '*.incomplete' 2>/dev/null | wc -l)" -eq 0 ] || return 1
  $PY - "$MODEL" <<'PYEOF'
import json, os, sys
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
for f in sorted(set(idx["weight_map"].values())):
    if not os.path.exists(os.path.join(d, f)):
        sys.exit(1)
PYEOF
}

downloader_alive () {
  # match the script path, not the pattern we are running under
  pgrep -f "python $DL" >/dev/null 2>&1
}

start_downloader () {
  log "starting downloader"
  ( cd /physis/unified-dopsd && HF_ENDPOINT=https://hf-mirror.com \
      nohup "$PY" "$DL" >> /physis/unified-dopsd/qwen3vl_dl_restart.log 2>&1 & )
  sleep 15
}

log "supervisor up"
while ! weights_ready; do
  downloader_alive || start_downloader
  sleep 60
done

log "weights ready ($(du -sh "$MODEL" | cut -f1)) -- launching scoring"
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT"
bash scripts/score_geneval2.sh >> "$LOG" 2>&1
log "SCORING DONE"
