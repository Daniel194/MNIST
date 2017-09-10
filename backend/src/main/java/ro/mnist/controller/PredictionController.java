package ro.mnist.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import ro.mnist.entity.Prediction;
import ro.mnist.service.PredictionService;

import java.io.IOException;
import java.util.List;

@RestController
public class PredictionController {

    private PredictionService predictionService;

    @Autowired
    public PredictionController(PredictionService predictionService) {
        this.predictionService = predictionService;
    }

    @RequestMapping(value = "/api/prediction", method = RequestMethod.POST)
    @ResponseBody
    public List<Prediction> makePrediction(@RequestBody String imageBase64) throws IOException {
        return predictionService.makePrediction(imageBase64);
    }

}