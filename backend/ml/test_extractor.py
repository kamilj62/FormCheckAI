from app.main import extract_video_biomechanics

video = "/Users/josephkamil/Desktop/Capstone/Oly_Data/raw/snatch_mp4/1DX0IXizcHg_02062_02337.mp4"

sequence, biomechanics, debug = extract_video_biomechanics(video)

print("Sequence frames:", len(sequence))
print("Biomechanics frames:", len(biomechanics))
print(debug)

if biomechanics:
    print("\nKeys:")
    print(sorted(biomechanics[0].keys())[:20])