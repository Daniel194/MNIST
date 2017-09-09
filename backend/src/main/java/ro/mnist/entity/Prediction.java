package ro.mnist.entity;

public class Prediction {

    private Integer nr;
    private Float accuracy;

    public Integer getNr() {
        return nr;
    }

    public void setNr(Integer nr) {
        this.nr = nr;
    }

    public Float getAccuracy() {
        return accuracy;
    }

    public void setAccuracy(Float accuracy) {
        this.accuracy = accuracy;
    }
}
