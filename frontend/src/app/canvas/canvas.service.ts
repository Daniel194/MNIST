import {Injectable} from '@angular/core';
import {Http, Response} from '@angular/http';
import {Headers, RequestOptions} from '@angular/http';
import 'rxjs/add/operator/map';
import 'rxjs/add/operator/toPromise';

import {Prediction} from './Prediction';


@Injectable()
export class CanvasService {
    private canvasPredictionUrl = 'http://localhost:8080/api/prediction';
    private headers = new Headers({'Content-Type': 'application/json'});
    private options = new RequestOptions({headers: this.headers});

    constructor(private http: Http) {
    }


    public makePrediction(image: ImageData): Promise<Prediction[]> {
        return this.http.post(this.canvasPredictionUrl, image, this.options).toPromise()
            .then(this.extractData)
            .catch(this.handleError);
    }

    private handleError(error: any): Promise<any> {
        return Promise.reject(error.message || error);
    }

    private extractData(res: Response) {
        let body = res.json();
        return body.data || {};
    }

}