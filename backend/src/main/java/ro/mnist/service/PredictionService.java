package ro.mnist.service;

import javassist.bytecode.stackmap.TypeData;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import ro.mnist.entity.Prediction;
import sun.misc.BASE64Decoder;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

@Service
public class PredictionService {

    private static final Logger LOGGER = Logger.getLogger(TypeData.ClassName.class.getName());

    @Value("${image.folder}")
    private String imageFolder;

    public List<Prediction> makePrediction(String data) {
        String base64Image = data.split(",")[1];
        BufferedImage image = null;
        byte[] imageByte;
        BASE64Decoder decoder = new BASE64Decoder();

        try {
            imageByte = decoder.decodeBuffer(base64Image);


            ByteArrayInputStream bis = new ByteArrayInputStream(imageByte);
            image = ImageIO.read(bis);
            bis.close();

            // write the image to a file
            File outputfile = new File(imageFolder + "image.png");
            ImageIO.write(image, "png", outputfile);

        } catch (IOException ex) {
            LOGGER.log(Level.SEVERE, ex.toString(), ex);
        }

        return null;
    }

}
