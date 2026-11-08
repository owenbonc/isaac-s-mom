#!/bin/zsh
# Presenter launcher. Each mode plays the simulation in a window while it runs, and saves an MP4 to videos/.
#   ./run.sh grey    the same scene twice: with its camera, then with a grey image instead (about 30 s)
#   ./run.sh live    the student alone, bin moved after the lift, with opening and closing cards (about 40 s)
#   ./run.sh moved   a second scene with the bin moved, no cards
#   ./run.sh video   re-film the simulation video at 0.5x speed (file only)
# Needs the showcase checkpoint in runs/ (see README.md). Set PY to use another interpreter.
cd "${0:A:h}"
PY=(${=PY:-uv run --no-sync python})
T=$(date +%H%M%S)
mkdir -p videos
case "${1:-grey}" in
  grey)  exec $PY sim_video2.py --seed 900301 --pair --live --speed 1 --out videos/grey-$T.mp4 ;;
  live)  exec $PY sim_video2.py --seed 900407 --live --speed 1 --out videos/live-$T.mp4 ;;
  moved) exec $PY sim_video2.py --seed 900409 --live --speed 1 --out videos/moved-$T.mp4 ;;
  video) exec $PY sim_video2.py --seed 900407 --out videos/scaffld-simulation-$T.mp4 ;;
  *)     echo "usage: ./run.sh grey | live | moved | video"; exit 1 ;;
esac
