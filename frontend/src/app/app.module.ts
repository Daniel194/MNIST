import { NgModule }      from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppComponent }  from './app.component';
import { CanvasComponent } from './canvas/canvas.component';

@NgModule({
    imports:      [ BrowserModule ],
    declarations: [ AppComponent, CanvasComponent ],
    bootstrap:    [ AppComponent ]
})
export class AppModule { }