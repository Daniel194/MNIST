import {
    Component, Input, ElementRef, AfterViewInit, ViewChild
} from '@angular/core';
import {Observable} from 'rxjs/Observable';

import 'rxjs/add/observable/fromEvent';
import 'rxjs/add/operator/takeUntil';
import 'rxjs/add/operator/pairwise';
import 'rxjs/add/operator/switchMap';
import {CanvasService} from './canvas.service';
import {Prediction} from './Prediction';

@Component({
    selector: 'app-canvas',
    templateUrl: './canvas.template.html',
    styleUrls: ['./canvas.style.css'],
    providers: [CanvasService]
})
export class CanvasComponent implements AfterViewInit {

    @ViewChild('canvas') public canvas: ElementRef;

    @Input() public width = 400;
    @Input() public height = 400;

    @Input() public pictureSrc = '';

    public predictions: Prediction[] = [];

    private cx: CanvasRenderingContext2D;

    constructor(private canvasService: CanvasService) {
        this.initializePrediction();
    }

    public ngAfterViewInit() {
        const canvasEl: HTMLCanvasElement = this.canvas.nativeElement;
        this.cx = canvasEl.getContext('2d');

        canvasEl.width = this.width;
        canvasEl.height = this.height;

        this.cx.lineWidth = 3;
        this.cx.lineCap = 'round';
        this.cx.strokeStyle = '#000';

        this.captureEvents(canvasEl);
        this.pictureSrc = this.canvas.nativeElement.toDataURL();
    }

    public clear(event: MouseEvent) {
        this.cx.clearRect(0, 0, this.width, this.height);
    }

    public predict(event: MouseEvent) {
        this.pictureSrc = this.canvas.nativeElement.toDataURL();
        this.canvasService.makePrediction(this.cx.getImageData(0, 0, this.width, this.height))
            .then(function (value) {
                this.predictions = value;
            }, function (reason) {
                this.initializePrediction();
            });
    }

    private captureEvents(canvasEl: HTMLCanvasElement) {
        Observable
            .fromEvent(canvasEl, 'mousedown')
            .switchMap((e) => {
                return Observable
                    .fromEvent(canvasEl, 'mousemove')
                    .takeUntil(Observable.fromEvent(canvasEl, 'mouseup'))
                    .pairwise()
            })
            .subscribe((res: [MouseEvent, MouseEvent]) => {
                const rect = canvasEl.getBoundingClientRect();

                const prevPos = {
                    x: res[0].clientX - rect.left,
                    y: res[0].clientY - rect.top
                };

                const currentPos = {
                    x: res[1].clientX - rect.left,
                    y: res[1].clientY - rect.top
                };

                this.drawOnCanvas(prevPos, currentPos);
            });
    }

    private drawOnCanvas(prevPos: { x: number, y: number }, currentPos: { x: number, y: number }) {
        if (!this.cx) {
            return;
        }

        this.cx.beginPath();

        if (prevPos) {
            this.cx.moveTo(prevPos.x, prevPos.y); // from
            this.cx.lineTo(currentPos.x, currentPos.y);
            this.cx.stroke();
        }
    }

    private initializePrediction() {
        for (let i = 0; i < 10; i++) {
            this.predictions.push({nr: i, accuracy: 0.0});
        }
    }
}