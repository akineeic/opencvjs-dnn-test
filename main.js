let testInfo = document.getElementById("testInfo");
let iterationProgress = document.getElementById("iterationProgress");
let showIterations = document.getElementById("showIterations");
let modelLoad = document.getElementById("modelLoad");
let modelProgress = document.getElementById("modelProgress");

let runButton = document.getElementById("runButton")
runButton.addEventListener("click", run)

let imgelement = document.getElementById("imgsrc");
let inputelement = document.getElementById("fileinput");
inputelement.addEventListener("change", (e) =>{
    imgelement.src = URL.createObjectURL(e.target.files[0]);
}, false)

let modelStatus = ['squeezenet', 'mobilenetv2', 'resnet50v1', 'resnet50v2'];
let iterations = Number(document.querySelector('#iterations').value);
let initFlag = true, calIteration = 0, topNum = 5, timeSum = [], labels, classes;

function run(){
    initPara();
    console.log('load model...')
    let onnxmodel = document.getElementById("modelName").value;
    index = modelStatus.indexOf(onnxmodel);
    if ( index != -1){
        onnxmodel += '.onnx';
        createFileFromUrl(onnxmodel, onnxmodel, compute);
        modelStatus.splice(index, 1);
    } else{
        modelLoad.innerHTML = `${onnxmodel} has been loaded before.`
        compute();
    };
}

function loadLables(){
    let url = 'labels1000.txt';
    let request = new XMLHttpRequest();
    request.open('GET', url, true);
    request.onload = function(ev) {
        if (request.readyState ===4 ) {
            if(request.status === 200) {
                labels = request.response;
                labels = labels.split('\n')
            };
        };
    };
    request.send();
};

function compute (){
    let inputMat = imageToMat();
    let onnxmodel = document.getElementById("modelName").value + '.onnx';
    let net = cv.readNetFromONNX(onnxmodel);
    console.log('Start inference...')
    let input = cv.blobFromImage(inputMat, 1, new cv.Size(224, 224), new cv.Scalar(0,0,0));
    net.setInput(input);

    eachForward(net);
}

function eachForward(net){
    let start = performance.now();
    let result = net.forward();
    let end = performance.now();

    classes = getTopClasses(result);
    let delta = end - start;
    console.log(`Iterations: ${calIteration+1} / ${iterations+1}, inference time: ${delta}ms`);
    printResult(classes);
    timeSum.push(delta);   
    iterationState();
    ++calIteration;

    if(calIteration<iterations+1){
        setTimeout(function(){
            eachForward(net);
        }, 0);
    } else{
        console.log('Test finished!');
        updateResult(classes);
    }
}

function imageToMat(){
    let mat = cv.imread("imgsrc");
    let matC3 = new cv.Mat(mat.matSize[0],mat.matSize[1],cv.CV_8UC3);
    cv.cvtColor(mat, matC3, cv.COLOR_RGBA2RGB);
    let matdata = matC3.data;
    let stddata = [];
    for(var i=0; i<mat.matSize[0]*mat.matSize[1]; ++i){
        stddata.push( (matdata[3*i]/255-0.485)/0.229 ); 
        stddata.push( (matdata[3*i+1]/255-0.456)/0.224 ); 
        stddata.push( (matdata[3*i+2]/255-0.406)/0.225 );
    };
    let inputMat = cv.matFromArray(mat.matSize[0],mat.matSize[1],cv.CV_32FC3,stddata);

    return inputMat;
}

function iterationState() {
    iterationProgress.style.visibility = 'visible';
    iterationProgress.value = (calIteration+1)*100/(iterations+1);
    showIterations.innerHTML = `Iterations: ${calIteration+1} / ${iterations+1}`;
}

function updateResult(classes){
    let finalResult = summarize(timeSum);
    testInfo.style.visibility="visible";
    testInfo.innerHTML = `<b>Build type</b>: ${document.getElementById("title").innerHTML.split(/[()]/)[1]} <br>
                                                    <b>Model</b>: ${document.getElementById("modelName").value} <br>
                                                    <b>Inference Time</b>: ${finalResult.mean.toFixed(2)}`;
    if(iterations != 1){
        testInfo.innerHTML += ` ± ${finalResult.std.toFixed(2)} [ms] <br> <br>`;
    } else{
        testInfo.innerHTML += ` [ms] <br> <br>`;
    };
    testInfo.innerHTML += `<b>label1</b>: ${classes[0].label}, probability: ${classes[0].prob}% <br>
                           <b>label2</b>: ${classes[1].label}, probability: ${classes[1].prob}% <br>
                           <b>label3</b>: ${classes[2].label}, probability: ${classes[2].prob}% <br>
                           <b>label4</b>: ${classes[3].label}, probability: ${classes[3].prob}% <br>
                           <b>label5</b>: ${classes[4].label}, probability: ${classes[4].prob}%` ;
}

function initPara(){
    iterations = Number(document.querySelector('#iterations').value);
    calIteration = 0;
    timeSum = [];

    testInfo.innerHTML = '';
    modelLoad.innerHTML = '';
    modelProgress.value = 0;
    modelProgress.style.visibility = 'hidden';
    iterationProgress.style.visibility = 'hidden';
    showIterations.innerHTML = '';

    if(initFlag){
        loadLables();
        initFlag = false;
    };
}

function createFileFromUrl(path, url, callback){
    let request = new XMLHttpRequest();
    request.open('GET', url, true);
    request.responseType = 'arraybuffer';
    request.onload = function(ev) {
        if (request.readyState === 4) {
            if (request.status === 200) {
                let data = new Uint8Array(request.response);
                cv.FS_createDataFile('/', path, data, true, false, false);
                callback();
            } else {
                console.log('Failed to load ' + url + ' status: ' + request.status);
            }
        }
    };
    request.send();
    request.onprogress = modelState;
};

function modelState(ev){
    modelProgress.style.visibility = 'visible';
    let totalSize = ev.total / (1000 * 1000);
    let loadedSize = ev.loaded / (1000 * 1000);
    let percentComplete = ev.loaded / ev.total * 100;
    modelLoad.innerHTML = `${loadedSize.toFixed(2)}/${totalSize.toFixed(2)}MB ${percentComplete.toFixed(2)}%`;
    modelProgress.value = percentComplete;
}

function softmax(arr) {
    const C = Math.max(...arr);
    const d = arr.map((y) => Math.exp(y - C)).reduce((a, b) => a + b);
    return arr.map((value, index) => { 
        return Math.exp(value - C) / d;
    })
}


function getTopClasses(mat) {
    let initdata = mat.data32F;
    initdata = softmax(initdata);
    let probs = Array.from(initdata);
    let indexes = probs.map((prob, index) => [prob, index]);
    let sorted = indexes.sort((a, b) => {
    if (a[0] === b[0]) {return 0;}
    return a[0] < b[0] ? -1 : 1;
    });
    sorted.reverse();
    let classes = [];
    for (let i = 0; i < topNum; ++i) {
    let prob = sorted[i][0];
    let index = sorted[i][1];
    let c = {
        label: labels[index],
        prob: (prob * 100).toFixed(2)
    }
    classes.push(c);
    }
    return classes;
}

function printResult(classes){
    for (let i = 0; i < topNum; ++i){
        console.log(`label: ${classes[i].label}, probability: ${classes[i].prob}%`)
    }
}

function summarize(results) {
    if (results.length !== 0) {
        // remove first run, which is regarded as "warming up" execution
        results.shift();
        let d = results.reduce((d, v) => {
            d.sum += v;
            d.sum2 += v * v;
            return d;
        }, {
            sum: 0,
            sum2: 0
        });
        let mean = d.sum / results.length;
        let std = Math.sqrt((d.sum2 - results.length * mean * mean) / (results.length - 1));
        return {
            mean: mean,
            std: std
        };
    } else {
        return null;
    }
}
