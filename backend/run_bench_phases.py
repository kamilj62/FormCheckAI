from app.main import create_bench_press_phase_images, OVERLAY_DIR

phase_images = create_bench_press_phase_images(
    "/Users/josephkamil/Desktop/Capstone/IdealBenchPress.mov",
    OVERLAY_DIR,
    {
        "start_frame": 0,
        "end_frame": 220,
    },
    sample_every=10,
)

print("PHASE IMAGES:")
print(phase_images)