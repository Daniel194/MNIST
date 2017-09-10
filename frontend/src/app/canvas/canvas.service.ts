import {Injectable} from '@angular/core';
import {Http, Response} from '@angular/http';
import {Headers, RequestOptions} from '@angular/http';
import 'rxjs/add/operator/map';
import 'rxjs/add/operator/toPromise';

import {Prediction} from './Prediction';
import {Observable} from 'rxjs/Rx';


@Injectable()
export class CanvasService {
    private canvasPredictionUrl = 'http://localhost:8080/api/prediction';
    private headers = new Headers({'Content-Type': 'application/json'});
    private options = new RequestOptions({headers: this.headers});

    constructor(private http: Http) {
    }

    public makePrediction(canvas: HTMLCanvasElement): Observable<Prediction[]> {
        return this.http.post(this.canvasPredictionUrl, canvas.toDataURL('image/png'), this.options)
            .map((res: Response) => res.json())
            .catch((error: any) => Observable.throw(error.json().error || 'Server error'));
    }

}