
let data= [], label;

let model;

let guess_tf, input_tf;


const image_size= 28*28;
const LEARNING_RATE= 0.4;

const data_length= 30000; // should be > batchsize and < 60,000;


const totalEpoch= 4;
const batchSize= 50;

let trained= false;
let batchCount= 0, epochCount= 0;

function preload() {
    loadBytes("data/train-images.idx3-ubyte", (raw_data) => {
        let training_size= 16+ (data_length* image_size) ;
        for(let i=16; i<training_size; i+=image_size) {
            let tmp_array= Array.from(raw_data.bytes.slice(i, i+image_size)).map(x => x/255);
            data.push(tmp_array);
        }

        //  for(let i=16; i<raw_data.bytes.length; i+=image_size) {
        //     data.push(raw_data.bytes.slice(i, i+image_size).map(x => x/255));
        //  }
    });
    loadBytes("data/train-labels.idx1-ubyte", (raw_label) => {
        label= raw_label.bytes.slice(8, data_length+8); //first 8 bytes are headers
    });
}


function setup() {

    createCanvas(280, 280);
    background(100);
    
    for(let i=0; i<10; i++) {
        for(let j=0; j<10; j++) {

            let img= makeImgFromArray(data[10*i + j]);

            let y= i*28;
            let x= j*28;
            image(img, x, y);
        }
    }

    input_tf= tf.tensor2d(data);
    let guess_tf_arr= tf.tensor1d(label, 'int32');
    guess_tf = tf.oneHot(guess_tf_arr, 10).cast('float32');
    //guess_tf_arr.dispose();


    model= tf.sequential();

    model.add(tf.layers.dense({
        units: 1500,
        inputShape: [784],
        activation: 'sigmoid'
    }));

    model.add(tf.layers.dense({
        units: 500,
        activation: 'sigmoid'
    }));

    model.add(tf.layers.dense({
        units: 10,
        activation: 'softmax'
    }));


    const optimizer = tf.train.adam(LEARNING_RATE);

    model.compile({
        optimizer,
        //loss: 'categoricalCrossentropy',
        loss: 'meanSquaredError',
        //metrics: ['accuracy']
    });

    train();
};



async function train() {
    await model.fit(input_tf, guess_tf, {
        shuffle: true,
        validationSplit: 0.1,
        epochs: totalEpoch,
        batchSize: batchSize,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log("epoch:" + epoch + " & loss: " + logs.loss.toFixed(5));
                epochCount= epoch+1;
            },
            onBatchEnd: async (batch, logs) => {
                //console.log(" trained: "+ (batch+1)+ " & loss: " + logs.loss.toFixed(5));
                batchCount= batch;
                await tf.nextFrame();
            },
            onTrainEnd: () => {
                console.log('finished');
                trained= true;
                background(0);
            },
        },
    });
};



function makeImgFromArray(arr) {
    let img= createImage(28, 28);
    img.loadPixels();

    for(let k=0; k<image_size; k++) {
        let val= Math.floor(arr[k]*255);
        img.pixels[k*4 + 0]= val;
        img.pixels[k*4 + 1]= val;
        img.pixels[k*4 + 2]= val;
        img.pixels[k*4 + 3]= 255;
    }
    img.updatePixels();
    return img;
}

function draw() {
    if(!trained) {
        show_trained_percentage();
    }
    else {
        strokeWeight(30);
        stroke(255);
        if(mouseIsPressed) {
            line(pmouseX, pmouseY, mouseX, mouseY);
        }
    }
}


function show_trained_percentage() {
    background(51);
    let percentage= 100* ((data_length*0.9*epochCount)+(batchCount*batchSize))/(data_length*0.9*totalEpoch);
    textSize(24);
    textAlign(CENTER, CENTER);
    noStroke();
    fill(255);
    text("Training: "+percentage.toFixed(1)+ "%", 140, 140);
}


document.getElementById("_clear").addEventListener("click", () => {
    background(0);
}, false);

document.getElementById("guess").addEventListener("click", function() {
    let input= [];
    let img= get();
    img.resize(28, 28);
    img.loadPixels();
    for(let i=0; i<image_size; i++) {
        let bright= img.pixels[i*4];
        input[i]= bright/255.0;
    };
    //console.log(input);
    guess(input, (res) => {
        document.getElementById("result").innerText= "Guess: "+res;
    });
}, false);


let guess= (ip_arr, callback) => {
    tf.tidy(() => {
        let ip_tf= tf.tensor2d([ip_arr]);

        console.log("ip_arr=>\n\t"+ip_arr);

        let results = model.predict(ip_tf);
        let argMax = results.argMax(1);
        let guess = argMax.dataSync()[0];


        console.log(guess);
        if(callback) {
            callback(guess);
        }
    });
};

// function get() {
//     let input= [];
//     let img= get();
//     img.resize(28, 28);
//     img.loadPixels();
//     for(let i=0; i<image_size; i++) {
//         let bright= img.pixels[i*4];
//         input[i]= bright/255.0;
//     };
//     tf.tidy(() => {
//         let ip_tf= tf.tensor2d([arr])
//         let guess = model.predict(ip_tf).dataSync();
//         ip_tf.dispose();
//         console.table(guess);
//         return guess.indexOf(1);
//     });
// }