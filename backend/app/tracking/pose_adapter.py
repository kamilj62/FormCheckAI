def remap_crop_landmarks_to_full_frame(results, crop_box, frame_width, frame_height):
    if not results.pose_landmarks:
        return results

    x1, y1, x2, y2 = crop_box
    crop_w = x2 - x1
    crop_h = y2 - y1

    for lm in results.pose_landmarks.landmark:
        lm.x = (x1 + lm.x * crop_w) / frame_width
        lm.y = (y1 + lm.y * crop_h) / frame_height

    return results