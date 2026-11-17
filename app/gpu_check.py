"""Run inside the GPU worker: python -m app.gpu_check (no model download)."""

import ctypes
import platform


def main():
    import ctranslate2

    count = ctranslate2.get_cuda_device_count()
    print(f"CTranslate2 {ctranslate2.__version__}; CUDA devices: {count}")
    if not count:
        raise RuntimeError("GPU не виден. Запустите Docker с docker-compose.gpu.yml и WSL2.")
    if platform.system() == "Linux":
        # Match CTranslate2 4.4 from uv.lock; check actual dynamic-library loading.
        for library in ("libcublas.so.12", "libcudnn.so.8"):
            ctypes.CDLL(library)
            print(f"Loaded: {library}")
    types = ctranslate2.get_supported_compute_types("cuda")
    print("CUDA compute types: " + ", ".join(sorted(types)))
    if "float16" not in types:
        raise RuntimeError("GPU не поддерживает выбранный float16.")
    print("CUDA runtime check OK. Actual model inference is checked on the next audio job.")


if __name__ == "__main__":
    main()
