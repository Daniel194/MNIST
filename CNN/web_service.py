from flask import Flask
from flask_restful import Resource, Api
from CNN import MNIST
import numpy as np
from PIL import Image

app = Flask(__name__)
api = Api(app)


class Prediction(Resource):
    def get(self):
        model = MNIST()

        img = Image.open('/home/ldaniel/Desktop/MNIST/backend/src/main/resources/image/image.png')
        img.thumbnail((28, 28), Image.ANTIALIAS)

        pix = np.array(img)
        pix = pix[:, :, 3]

        pred = model.predict(pix.reshape(1, 1, 28, 28))

        return {'prediction': pred.tolist()}


api.add_resource(Prediction, '/prediction')

if __name__ == '__main__':
    app.run(port='5000')
