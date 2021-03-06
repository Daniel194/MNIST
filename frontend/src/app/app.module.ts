import { NgModule }      from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppComponent }  from './app.component';
import { CanvasComponent } from './canvas/canvas.component';
import {HttpModule} from '@angular/http';

@NgModule({
    imports:      [ BrowserModule, HttpModule],
    declarations: [ AppComponent, CanvasComponent ],
    bootstrap:    [ AppComponent ]
})
export class AppModule { }