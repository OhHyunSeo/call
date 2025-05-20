import librosa

def test_librosa_load():
    test_file = "test.wav"

    try:
        print(f"[INFO] Trying to load: {test_file}")
        y, sr = librosa.load(test_file, sr=None)
        print(f"[SUCCESS] Loaded file. Sample rate: {sr}, length: {len(y)} samples.")
    except Exception as e:
        print(f"[ERROR] Failed to load audio with librosa: {e}")

if __name__ == "__main__":
    test_librosa_load()
