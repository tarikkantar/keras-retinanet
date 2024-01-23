from fastapi import FastAPI, File, UploadFile
import uvicorn 
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf 
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image

app = FastAPI()

model = models.load_model("../savedModel/resnet50_csv_44.h5", backbone_name='resnet50')
MODEL = models.convert_model(model)

CLASS_NAMES = ['demir dikeni zehirli', 'gelincik zehirli',
       'guzelavrat otu zehirli', 'hazeran zehirli', 'kanarya otu zehirli',
       'kanavci otu zehirli', 'karamuk zehirli', 'porsuk otu zehirli',
       'yabani salgam zehirli', 'yuksuk otu zehirli',
       'sari tas yoncasi zehirsiz', 'sorgum zehirsiz',
       'soya fasulyesi zehirsiz', 'tarla sarmasigi zehirsiz',
       'yonca zehirsiz', 'burcak zehirsiz', 'cilek ucgulu zehirsiz',
       'hayvan pancari zehirsiz', 'kirmizi ucgul zehirsiz',
       'korunga zehirsiz']

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    try:
        print(file)
        im = np.array(Image.open(BytesIO(await file.read())))
        im = im[:, :, :3]
        imp = preprocess_image(im)
        imp, scale = resize_image(im)
        boxes, scores, labels = MODEL.predict_on_batch(np.expand_dims(imp, axis=0))
        boxes /= scale

        predictions = []
        for score, label in zip(scores[0], labels[0]):
            if score > 0.5:
                predictions.append({
                    'class': CLASS_NAMES[label],
                    'confidence': float(score)
                })
        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)