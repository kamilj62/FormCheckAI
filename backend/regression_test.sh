#!/bin/bash

API="http://localhost:8000/analyze"

run_test () {
  FILE=$1
  LABEL=$2

  echo "=============================="
  echo "TEST: $LABEL"
  echo "FILE: $FILE"
  echo "=============================="

  curl -s -X POST \
    -F "file=@$FILE" \
    $API | jq "{
      expected: \"$LABEL\",
      predicted: .exercise_label,
      confidence: .confidence,
      mode: .analysis_mode
    }"

  echo ""
}

# =========================
# SQUATS
# =========================
run_test "/Users/josephkamil/Desktop/Capstone/backsquat_short.mov" "squat_back"
run_test "/Users/josephkamil/Desktop/Capstone/FrontSquat- correct3.mov" "squat_front"
run_test "/Users/josephkamil/Desktop/Capstone/OverheadSquat_one_rep_720p.mp4" "overhead_squat"

# =========================
# OLYMPIC LIFTS
# =========================
run_test "/Users/josephkamil/Desktop/Capstone/CleanAndJerk/v_CleanAndJerk_g25_c04.avi" "clean_and_jerk"
run_test "/Users/josephkamil/Desktop/Capstone/snatch- correct.mov" "snatch"
run_test "/Users/josephkamil/Desktop/Capstone/splitjerk-short.mp4" "split_jerk"
run_test "/Users/josephkamil/Desktop/Capstone/clean-correct.mov" "clean"

# =========================
# EDGE CASES
# =========================
run_test "/Users/josephkamil/Desktop/Capstone/deadlift/deadlift_1.mp4" "deadlift"
run_test "/Users/josephkamil/Desktop/Capstone/pushpress_short.mov" "push_press"
run_test "/Users/josephkamil/Desktop/Capstone/bench_short.mov" "bench_press"

# =========================
# NEW LIFTS (ADDED)
# =========================

run_test "/Users/josephkamil/Desktop/Capstone/thruster-correct-small.mp4" "thruster"
run_test "/Users/josephkamil/Desktop/Capstone/strict press.mov" "strict_press"