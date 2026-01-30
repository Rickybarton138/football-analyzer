"""Extract first 30 minutes from video."""
from moviepy.editor import VideoFileClip
import sys

input_path = r"C:\Users\info\Downloads\balti-away-2026-2026-01-10.mp4"
output_path = r"C:\Users\info\football-analyzer\backend\uploads\balti-away-30min.mp4"

print(f"Loading video: {input_path}")
video = VideoFileClip(input_path)

duration = video.duration
print(f"Total duration: {duration/60:.1f} minutes")

# Extract first 30 minutes (1800 seconds)
clip_duration = min(1800, duration)
print(f"Extracting first {clip_duration/60:.1f} minutes...")

clip = video.subclip(0, clip_duration)

print(f"Saving to: {output_path}")
clip.write_videofile(
    output_path,
    codec='libx264',
    audio_codec='aac',
    temp_audiofile='temp-audio.m4a',
    remove_temp=True,
    verbose=False,
    logger=None
)

video.close()
clip.close()

print(f"Done! Extracted clip saved to: {output_path}")
