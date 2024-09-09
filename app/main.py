import modal


app = modal.App("surakshita")
image = (
    modal.Image.debian_slim()
    .apt_install("fonts-freefont-ttf tesseract-ocr")
    .pip_install(
        "huggingface-hub==0.16.4",
        "Pillow",
        "timm",
        "transformers",
        "acclerate",
        "pytesseract",
        "pandas",
        "numpy,
        "opencv-contrib-python",
        "pillow",
    )
)