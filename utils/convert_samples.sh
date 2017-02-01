#!/bin/sh

# this script converts all mp3 files within the working directory to 22050Hz
# wav files. A filename format of ID\ <text>.mp3 is assumed.

FILES=$(find . -name '*.mp3')
while read -r f
do
  path=$(dirname "$f")
  id=$(basename "$f" | cut -d' ' -f 1 | cut -d'.' -f 1)
  nf="$path/$id.wav"
  echo "[$id] $f -> $nf"
  if [ -f "$nf" ]; then
    echo "  exists"
  else
    eval ffmpeg -loglevel error -i \"${f}\" -ar 22050 \"${nf}\"
  fi
done <<< "$FILES"

#for f in ./*.mp3
#do
#  ID=$(echo ${f} | cut -d' ' -f 1 | cut -d'/' -f 2 | cut -d'.' -f 1)
#  echo "converting ${ID}..."
#  eval ffmpeg -loglevel panic -i \"$f\" -ar 22050 ${ID}.wav
#done
