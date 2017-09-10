package ro.mnist.service;

import org.json.JSONArray;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import ro.mnist.aspect.ExceptionHandler;
import ro.mnist.entity.Prediction;
import sun.misc.BASE64Decoder;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

@Service
public class PredictionService {

    @Value("${image.folder}")
    private String imageFolder;

    @ExceptionHandler
    public List<Prediction> makePrediction(String data) throws IOException {
        writeImage(data);
        JSONArray jsonPrediction = new JSONArray(requestCNN());
        List<Prediction> predictions = new ArrayList<>();

        for (int i = 0; i < jsonPrediction.length(); i++) {
            predictions.add(new Prediction(i, jsonPrediction.getDouble(i)));
        }

        return predictions;
    }

    private void writeImage(String data) throws IOException {
        String base64Image = data.split(",")[1];
        BufferedImage image;
        byte[] imageByte;
        BASE64Decoder decoder = new BASE64Decoder();

        imageByte = decoder.decodeBuffer(base64Image);

        ByteArrayInputStream bis = new ByteArrayInputStream(imageByte);
        image = ImageIO.read(bis);
        bis.close();

        File outputfile = new File(imageFolder + "image.png");
        ImageIO.write(image, "png", outputfile);
    }

    private String requestCNN() throws IOException {
        URL url = new URL("http://127.0.0.1:5000/prediction");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("GET");
        conn.setRequestProperty("Accept", "application/json");

        if (conn.getResponseCode() == 200) {
            BufferedReader br = new BufferedReader(new InputStreamReader((conn.getInputStream())));
            return br.readLine();
        } else {
            return "";
        }
    }

}
