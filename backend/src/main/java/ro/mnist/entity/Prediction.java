package ro.mnist.entity;

public class Prediction {

    private Integer nr;
    private Double accuracy;

    public Prediction(Integer nr, Double accuracy) {
        this.nr = nr;
        this.accuracy = accuracy;
    }

    public Integer getNr() {
        return nr;
    }

    public void setNr(Integer nr) {
        this.nr = nr;
    }

    public Double getAccuracy() {
        return accuracy;
    }

    public void setAccuracy(Double accuracy) {
        this.accuracy = accuracy;
    }
}
