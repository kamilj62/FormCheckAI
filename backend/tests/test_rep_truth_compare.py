import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "ml"
    / "benchmark"
    / "compare_rep_truth.py"
)

spec = importlib.util.spec_from_file_location(
    "compare_rep_truth",
    MODULE_PATH,
)
compare_rep_truth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_rep_truth)

relative_video_key = compare_rep_truth.relative_video_key
result_keys = compare_rep_truth.result_keys


def test_relative_video_key_strips_capstone_root():
    assert relative_video_key(
        "/Users/josephkamil/Desktop/Capstone/thruster-correct.mov"
    ) == "thruster-correct.mov"


def test_result_keys_include_relative_path_and_name():
    keys = result_keys({
        "relative_path": "Oly_Data/segmented/clean/clean_from_cj_0012.mp4",
        "name": "clean_from_cj_0012.mp4",
    })

    assert "Oly_Data/segmented/clean/clean_from_cj_0012.mp4" in keys
    assert "clean_from_cj_0012.mp4" in keys
