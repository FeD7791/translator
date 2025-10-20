from faster_whisper import WhisperModel
import time
from tqdm import tqdm
import subprocess
import os
from datetime import datetime
import pathlib


def traducir(path_filename, sufix, device="cpu", model_size="large-v2"):

    path_filename = pathlib.Path(os.path.abspath(path_filename))

    records = pathlib.Path(path_filename.parent / "records")
    records.mkdir(exist_ok=True)

    transcriptions = pathlib.Path(path_filename.parent / "transcriptions")
    transcriptions.mkdir(exist_ok=True)


    # Load the model (try "large-v2" or "medium" for speed/accuracy balance)
    model = WhisperModel(model_size, device=device, compute_type="float16" if device=="cuda" else "int8")

    # Transform audio to .wav
    wavfile = _get_metadata(filepath=path_filename, sufix=sufix, format="wav")
    _transform_audio(input_file=path_filename, output_file=records / wavfile)

    # Transcribe + translate
    segments, info = model.transcribe(
        str(records / wavfile),
        task="transcribe",      # change to "transcribe" if you want original language
        language="es",
        beam_size=5,       # try 5–10
    	best_of=5
    )

    # Progress bar with total duration in seconds
    pbar = tqdm(total=info.duration, unit="sec", desc="Processing audio")

    full_text = []
    last_end = 0
    for segment in segments:
        full_text.append(segment.text)
        # advance progress bar
        pbar.update(segment.end - last_end)
        last_end = segment.end

    pbar.close()

    metadata = _get_metadata(filepath=path_filename, sufix=sufix, format="txt")
    # Write final text
    with open(transcriptions / metadata, "w", encoding="utf-8") as f:
        f.write(" ".join(full_text).strip())
        
 






# =============================================================================
# UTILS
# =============================================================================



def _transform_audio(input_file, output_file):
    subprocess.run([
        "ffmpeg",
        "-i", input_file,
        "-ar", "16000",   # resample to 16 kHz
        "-ac", "1",       # mono audio
        output_file
    ], check=True)

def _get_metadata(filepath, sufix, format):
    stat = os.stat(filepath)
    dt = datetime.fromtimestamp(stat.st_mtime)
    dt_dict = {
    "year": dt.year,
    "month": dt.month,
    "day": dt.day,
    "hour": dt.hour,
    "minute": dt.minute,
    "second": dt.second,
    }
    return f"{sufix}_{dt.day}_{dt.month}_{dt.year}.{format}"






# ffmpeg -i e_11_09_2025.aac -ar 16000 -ac 1 clean.wav
# traducir("./extragal29.aac", sufix="extragalactica", device="cpu", model_size="large-v2")
traducir("./lss13.aac", sufix="lss", device="cpu", model_size="large-v2")
