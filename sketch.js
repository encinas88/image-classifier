//local server** python -m http.server 8000 --bind 127.0.0.1 **
let video;
let labelP;
let features;
let knn;
let ready = false;
let label = "";

function setup(){
    createCanvas(400, 300);
    video=createCapture(VIDEO);
    video.size(400, 300);
    //mirrors image
    video.style("transform", "scale(-1,1)");
    //video.hide();
    features=ml5.featureExtractor('MobileNet', modelReady);
    labelP = createP("Please train me....");
    labelP.style('font-size','40pt');

    x = width  / 2;
    y = height / 2;
}

function goClassify(){
    const logits = features.infer(video);
    knn.classify(logits, function(error, result){
    if (error) {
        console.error(error);
    } else {
        label = result.label;
        labelP.html(label);
        goClassify();
    }
  });
}

function mousedPressed(){
}

// function keyPressed(){
//     const logits = features.infer(video);
//     if(key=="p"){
//         knn.addExample(logits, "Pikachu");
//         console.log('Pikachu');
//     } else if(key=="r"){
//         knn.addExample(logits, "Raichu");
//         console.log('Raichu');
//     } else if(key=="c"){
//         knn.addExample(logits, "Charmander");
//         console.log('Charmander');
//     } else if(key=="z"){
//         knn.addExample(logits, "Charizard");
//         console.log('Charizard');
//     } else if(key==" "){
//         knn.addExample(logits, "stay");
//     } else if(key=="s"){
//         knn.save("model.json");
//     }
//   }
function modelReady(){
    console.log('MobileNet connected');
        knn=ml5.KNNClassifier();
        knn.load('model.json', function (){
            console.log('KNN Model Loaded')
            goClassify();
        });
}


