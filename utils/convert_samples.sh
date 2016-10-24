#!/bin/sh

# this script converts all mp3 files within the working directory to 22050Hz
# wav files. A filename format of ID\ <text>.mp3 is assumed.

for f in ./*.mp3
do
  ID=$(echo ${f} | cut -d' ' -f 1 | cut -d'/' -f 2 | cut -d'.' -f 1)
  echo "converting ${ID}..."
  eval ffmpeg -loglevel panic -i \"$f\" -ar 22050 ${ID}.wav
done
