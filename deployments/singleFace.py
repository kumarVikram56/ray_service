from ray import serve
from fastapi import APIRouter, Response, UploadFile, status
import numpy as np
import cv2

from MLmodels.singleFace.embedding_gen import get_embedding_vector

router = APIRouter()

@serve.deployment
@serve.ingress(router)
class faceEmbedingGenModel:
    def __init__(self) -> None:
        pass

    @router.get("/face-embedding/")
    async def getFaceEmbeding(self, imageFile: UploadFile, response: Response):
        contents = await imageFile.read()
        toNpArr = np.frombuffer(contents, np.uint8)
        # Decode the NumPy array as an image using OpenCV
        img = cv2.imdecode(toNpArr, cv2.IMREAD_COLOR)
        try:
            emb_vector = get_embedding_vector(img)
            return {
                "vector_embeding": emb_vector
            }
        except Exception as err:
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return {
                "error": "Error in generating vector, Please make sure image have exact one Face!",
                "system_message": err
            }

getFaceEmbedingRouter = faceEmbedingGenModel.options(route_prefix="/").bind()