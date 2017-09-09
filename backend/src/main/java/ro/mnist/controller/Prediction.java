package ro.mnist.controller;

import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;

@RestController
public class Prediction {

    @RequestMapping(value = "/api/prediction", method = RequestMethod.POST)
    @ResponseBody
    public List<Prediction> makePrediction(@RequestBody String imageBase64) {

        List<Prediction> empty = new ArrayList<>();

        return empty;
    }

}