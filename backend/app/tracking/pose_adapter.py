def remap_crop_landmarks_to_full_frame(
    results,
    crop_box,
    frame_width,
    frame_height,
):
    if not results.pose_landmarks:
        return results

    x1, y1, x2, y2 = crop_box
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)

    frame_width = max(1, frame_width)
    frame_height = max(1, frame_height)

    for lm in results.pose_landmarks.landmark:
        # Convert crop-normalized coordinates to full-frame coordinates.
        lm.x = (x1 + lm.x * crop_w) / frame_width
        lm.y = (y1 + lm.y * crop_h) / frame_height

        # MediaPipe pose z is approximately normalized using image width.
        # Convert crop-relative depth to the same scale used by full-frame
        # inference so classifier inputs remain comparable.
        lm.z = lm.z * (crop_w / frame_width)

    return results
