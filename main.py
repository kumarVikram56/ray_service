from fastapi import FastAPI, File, UploadFile
from typing import Annotated
from ray import serve
import cv2
import numpy as np
from fastapi import status, Response
from MLmodels.singleFace.embedding_gen import get_embedding_vector
from dotenv import load_dotenv
import os


app = FastAPI()


@serve.deployment
@serve.ingress(app)
class faceEmbedingGenModel:
    def __init__(self) -> None:
        pass

    @app.get("/")
    def hello(self):
        return {"message": "hello from '/"}

    @app.post("/face-embedding/", status_code=200)
    async def getFaceEmbeding(
        self, raw_image: Annotated[UploadFile, File()], response: Response
    ):
        try:
            contents = await raw_image.read()
            toNpArr = np.frombuffer(contents, np.uint8)
            # Decode the NumPy array as an image using OpenCV
            img = cv2.imdecode(toNpArr, cv2.IMREAD_COLOR)
            getEmbVector = get_embedding_vector(img)
            embVector = getEmbVector.tolist()
        except TypeError:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"code": "NO_DETECTION", "message": "NO FACE FOUND IN IMAGE"}

        except Exception as err:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {
                "code": "INVALID_RAW_IMAGE",
                "message": "UNABE TO DECODE IMAGE FILE",
            }

        return {"vector_embeding": embVector}


# port = int(os.environ.get("PORT"))
serve.run(faceEmbedingGenModel.bind(), host="0.0.0.0", port=8080)
