import moviepy.editor as mp
clip = mp.VideoFileClip("video.mp4").subclip(0,15)
clip.audio.write_audiofile("theaudio.wav")
