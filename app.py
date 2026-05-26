from fastapi import FastAPI, UploadFile, File
import tempfile
import os

from inference import predict_paragraph


app = FastAPI()

ROLL_NO = "BSAI23034"


# ---------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "roll_no": ROLL_NO
    }


# ---------------------------------------------------
# OCR PREDICTION
# ---------------------------------------------------

@app.post("/predict")
async def predict(
    image: UploadFile = File(...)
):

    if not image.filename.lower().endswith(
        (
            ".png",
            ".jpg",
            ".jpeg",
        )
    ):

        return {
            "success": False,
            "error": "Invalid image"
        }

    temp_path = None

    try:

        contents = await image.read()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png",
        ) as temp:

            temp.write(contents)

            temp_path = temp.name

        prediction = predict_paragraph(
            temp_path
        )

        return {
            "success": True,
            "prediction": prediction
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

    finally:

        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)