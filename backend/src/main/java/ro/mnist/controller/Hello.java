package ro.mnist.controller;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class Hello {

    @RequestMapping("/api/hello")
    public String greet() {

        return "Hello from the other side!!!";

    }

}