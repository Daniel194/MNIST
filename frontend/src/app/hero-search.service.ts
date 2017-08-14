import {Injectable} from '@angular/core';
import {Http} from '@angular/http';

import {Observable} from 'rxjs/Observable';
import 'rxjs/add/operator/map';

import {Hero} from './hero';

@Injectable()
export class HeroSearchService {

    constructor(private http: Http) {
    }

    search(term: string): Observable<Hero[]> {
        return this.http
            .get(`http://localhost:8080/heroes/search/findByName?name=${term}`)
            .map(response => response.json()._embedded.heroes as Hero[]);
    }

}
