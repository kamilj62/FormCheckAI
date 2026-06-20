from app.celery_app import celery
from app.main import draw_overlay_video
import os

@celery.task
def generate_overlay(video_path, rep_feedback, label):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    return draw_overlay_video(
        video_path,
        f"/tmp/{label}.mp4",
        rep_feedback,
        label
    )