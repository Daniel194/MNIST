package ro.mnist.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import ro.mnist.entity.Prediction;
import ro.mnist.service.PredictionService;

import java.util.ArrayList;
import java.util.List;

@RestController
public class PredictionController {

    @Autowired
    private PredictionService predictionService;

    @RequestMapping(value = "/api/prediction", method = RequestMethod.POST)
    @ResponseBody
    public List<Prediction> makePrediction(@RequestBody String imageBase64) {

        List<Prediction> empty = predictionService.makePrediction(imageBase64);

        return empty;
    }

}