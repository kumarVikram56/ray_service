from fastapi import FastAPI
from ray import serve
import cv2
import numpy as np
from fastapi import UploadFile, status, Response
from MLmodels.singleFace.embedding_gen import get_embedding_vector
from dotenv import load_dotenv


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
    async def getFaceEmbeding(self, imageFile: UploadFile, response: Response):
        try:
            contents = await imageFile.read()
            toNpArr = np.frombuffer(contents, np.uint8)
            # Decode the NumPy array as an image using OpenCV
            img = cv2.imdecode(toNpArr, cv2.IMREAD_COLOR)
            emb_vector = get_embedding_vector(img).tolist()
        except TypeError:
            response.status_code = status.HTTP_404_NOT_FOUND
            return {"error": "No face found in image!"}
        except Exception as err:
            response.status_code = status.HTTP_406_NOT_ACCEPTABLE
            return {
                "error": "Error in generating vector, Please make sure image have exact one Face!",
                "system_message": err
            }

        return {
            "vector_embeding": emb_vector
        }


serve.run(faceEmbedingGenModel.bind(), host="0.0.0.0", port=8080)