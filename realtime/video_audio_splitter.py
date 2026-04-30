import os
import subprocess

def prepare_dataset(file_path):
    # 1. Verify the file exists
    if not os.path.exists(file_path):
        print(f"Error: Cannot find file at {file_path}")
        return
    
    if not file_path.lower().endswith(".mp4"):
        print("Error: The provided file is not an .mp4 video.")
        return

    # 2. Extract all the path pieces dynamically
    # Example: if file_path is "colDataPath/gayface/test.mp4"
    # base_dir = "colDataPath/gayface"
    # base_name = "test"
    base_dir = os.path.dirname(file_path)
    filename_with_ext = os.path.basename(file_path)
    base_name = os.path.splitext(filename_with_ext)[0]
    
    # 3. Create a new directory with the same name as the video in that exact location
    target_directory = os.path.join(base_dir, base_name)
    os.makedirs(target_directory, exist_ok=True)
    
    # 4. Define where the new video and audio will be saved
    target_video_path = os.path.join(target_directory, f"{base_name}.mp4")
    target_audio_path = os.path.join(target_directory, f"{base_name}.wav")

    print(f"Processing '{filename_with_ext}'...")

    # 5. Force Video to 25 FPS
    print(f"  Normalizing video to exactly 25 FPS -> {target_video_path}")
    vid_cmd = [
        "ffmpeg", "-y", "-i", file_path, 
        "-r", "25", # Force 25 FPS
        "-c:v", "libx264", "-preset", "fast", 
        target_video_path
    ]
    subprocess.run(vid_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # 6. Extract Audio to 16000Hz Mono
    print(f"  Extracting aligned audio -> {target_audio_path}")
    aud_cmd = [
        "ffmpeg", "-y", "-i", target_video_path, 
        "-vn", "-ac", "1", "-ar", "16000", 
        target_audio_path
    ]
    subprocess.run(aud_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    
    print(f"  Success! Processed files saved in {target_directory}\n")

if __name__ == "__main__":
    # Put your direct file path here
    videoName = "iamgay"
    file_path = f"test_video/{videoName}.mp4"
    
    prepare_dataset(file_path)