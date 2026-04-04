import os
import tempfile
import numpy as np
import soundfile as sf

print("TMPDIR env:", os.environ.get("TMPDIR"))
print("tempdir:", tempfile.gettempdir())

# 测试往临时目录写一个很小的 wav
test_wav = os.path.join(tempfile.gettempdir(), "test_write.wav")
test_audio = np.zeros(24000, dtype=np.float32)  # 1秒静音

try:
    sf.write(test_wav, test_audio, 24000)
    print("test write success:", test_wav)
except Exception as e:
    print("test write failed:", repr(e))