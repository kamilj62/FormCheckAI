#!/bin/bash

BASE_URL="http://127.0.0.1:8000/generate_visuals"
ROOT="/Users/josephkamil/Desktop/Capstone"

FILES=(
  "clean-correct.mov"
  "splitjerk-correct.mov"
  "thruster-correct.mov"
  "strictPullUp-correct.mov"
  "barMuscleUp.mov"
  "ringMuscleUp.mov"
)

echo "=============================="
echo "FORMCHECK SMOKE TEST"
echo "=============================="

for file in "${FILES[@]}"; do
  path="$ROOT/$file"

  echo ""
  echo "Testing: $file"

  curl -s -X POST "$BASE_URL" \
    -H "accept: application/json" \
    -F "file=@$path" \
  | python -m json.tool

  echo "------------------------------"
done

echo ""
echo "Done."