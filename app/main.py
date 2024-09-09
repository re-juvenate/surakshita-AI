import modal

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import UploadFile, File
from fastapi.responses import FileResponse

from app import functions
from app import db
from app import image_operations

router = APIRouter()

@router.post(path="/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Check if it's a PNG or JPEG image
    if file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(status_code=400, detail="Only PNG and JPEG images are allowed")

    image_uid = db.save_image(file)
    pii = functions.send_image(file, image_uid)

    return {"uid": image_uid, "pii": pii}


@router.post(path="/blur/")
async def blur_image(uid: str, pii: List[str]):
    image = db.get_image(uid)
    blurred_image = image_operations.apply_gaussian_blur(image)

    blurred_image_uid = db.save_image(blurred_image)

    # Perform some operations on the pii list
    processed_pii = functions.process_pii(pii)

    return {"uid": blurred_image_uid, "processed_pii": processed_pii}



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