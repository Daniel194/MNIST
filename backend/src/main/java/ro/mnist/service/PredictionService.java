package ro.mnist.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import ro.mnist.aspect.ExceptionHandler;
import ro.mnist.entity.Prediction;
import sun.misc.BASE64Decoder;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Service
public class PredictionService {

    @Value("${image.folder}")
    private String imageFolder;

    @ExceptionHandler
    public List<Prediction> makePrediction(String data) throws IOException {
        String base64Image = data.split(",")[1];
        BufferedImage image;
        byte[] imageByte;
        BASE64Decoder decoder = new BASE64Decoder();

        imageByte = decoder.decodeBuffer(base64Image);


        ByteArrayInputStream bis = new ByteArrayInputStream(imageByte);
        image = ImageIO.read(bis);
        bis.close();

        // write the image to a file
        File outputfile = new File(imageFolder + "image.png");
        ImageIO.write(image, "png", outputfile);

        return new ArrayList<>();
    }

}
