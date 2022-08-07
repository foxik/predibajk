#!/bin/bash

read header
echo -e "image_path\t$header\tcategory_short"

mkdir -p images

i=1
while read line; do
  path="images/$(printf %04d $i)"
  wget --quiet "$(echo "$line" | cut -f1)" -O $path || { echo Failed for image $i $line >&2; continue; }
  convert "$path" -fill white -opaque none -flatten -resize x480 -quality 97 "$path.jpg"
  echo -e "$path.jpg\t$line\t$(echo "$line" | sed 's/^.*\t//;s/-e*bike$//')"
  i=$(($i + 1))
done
