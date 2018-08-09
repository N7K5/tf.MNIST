
let raw_data, raw_label;

let image_size= 28*28;


function preload() {
    raw_data= loadBytes("data/train-images.idx3-ubyte");
    raw_label= loadBytes("data/train-labels.idx1-ubyte");
}


function setup() {

    createCanvas(280, 280);
    background(100);

    console.log(raw_data);
    
    for(let i=0; i<10; i++) {
        for(let j=0; j<10; j++) {
            let img= createImage(28, 28);
            img.loadPixels();

            let start_index= 16+ (image_size*((10*i)+j));

            for(let k=0; k<image_size; k++) {

                let val= raw_data.bytes[start_index+ k];
                
                img.pixels[k*4 + 0]= val;
                img.pixels[k*4 + 1]= val;
                img.pixels[k*4 + 2]= val;
                img.pixels[k*4 + 3]= 255;
            }
            img.updatePixels();
            let x= i*28;
            let y= j*28;
            image(img, x, y);
        }
    }

}