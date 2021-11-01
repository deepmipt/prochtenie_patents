#!/bin/sh

set -e

cd "$(dirname "$0")"
cd ..

echo "services:" > gpus.yml
for i in ` grep -E -i '^  [_a-z]*:[ #]*gpu' $1 | sed 's/:.*/:/g'`
do
  echo -e "  $i" >> gpus.yml
  echo -e "    environment:" >> gpus.yml
  echo -e '      - CUDA_VISIBLE_DEVICES=""' >> gpus.yml
done
echo 'version: "3.7"' >> gpus.yml