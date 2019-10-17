let runButton = document.getElementById("runButton")
runButton.addEventListener("click", run)

let imgelement = document.getElementById("imgsrc");
let inputelement = document.getElementById("fileinput");
inputelement.addEventListener("change", (e) =>{
    imgelement.src = URL.createObjectURL(e.target.files[0]);
}, false)

function run(){
    
    let iterations = Number(document.querySelector('#iterations').value);

    let net;
    let keywords;

    let mat = cv.imread("imgsrc");
    let matC3 = new cv.Mat(mat.matSize[0],mat.matSize[1],cv.CV_8UC3);
    cv.cvtColor(mat, matC3, cv.COLOR_RGBA2RGB);

    let matdata = matC3.data
    let stddata = []

    for(var i=0; i<mat.matSize[0]*mat.matSize[1]; ++i){
        stddata.push( (matdata[3*i]/255-0.485)/0.229 ); 
        stddata.push( (matdata[3*i+1]/255-0.456)/0.224 ); 
        stddata.push( (matdata[3*i+2]/255-0.406)/0.225 );
    }

    let newMat = cv.matFromArray(mat.matSize[0],mat.matSize[1],cv.CV_32FC3,stddata)

    console.log('load model...')
    let onnxmodel = document.getElementById("modelName").value + '.onnx';
    onnxUrl = "https://webnnmodel.s3-us-west-2.amazonaws.com/image_classification/model/squeezenet1.1.onnx";
        createFileFromUrl(onnxmodel, onnxmodel, () =>{
            let url = 'labels1000.txt';
            let request = new XMLHttpRequest();
            request.open('GET', url, true);
            request.onload = function(ev) {
                if (request.readyState ===4 ) {
                    if(request.status === 200) {
                    keywords = request.response;
                    keywords = keywords.split('\n')
                        net = cv.readNetFromONNX(onnxmodel);
                        console.log('start inference...')
                        let input = cv.blobFromImage(newMat, 1, new cv.Size(224, 224), new cv.Scalar(0,0,0));
                        net.setInput(input);
                        
                        let timeSum = [];
                        let topNum = 5;
                        for(let i = 0; i<iterations+1; ++i){
                            let start = performance.now();
                            var result = net.forward();
                            let end = performance.now();

                            let classes = getTopClasses(result, keywords, topNum);
                            let delta = end - start;
                            console.log(`Iterations: ${i+1} / ${iterations+1}, inference time: ${delta}ms`);
                            printResult(classes, topNum);
                            timeSum.push(delta);                                
                        };
                        let finalResult = summarize(timeSum);
                        console.log('Test finished!');
                        document.getElementById("result").style.visibility="visible";
                        document.getElementById("result").innerHTML = `Backend: ${document.getElementById("backend").value} <br>
                                                                        Model: ${document.getElementById("modelName").value} <br>
                                                                        Inference Time: ${finalResult.mean.toFixed(2)} `;
                        if(iterations != 1){
                            document.getElementById("result").innerHTML += `± ${finalResult.std.toFixed(2)} [ms]`
                        } else{
                            document.getElementById("result").innerHTML += `[ms]`
                        };
                    };
                };
            };
            request.send();
        });
};

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
};

function softmax(arr) {
    const C = Math.max(...arr);
    const d = arr.map((y) => Math.exp(y - C)).reduce((a, b) => a + b);
    return arr.map((value, index) => { 
        return Math.exp(value - C) / d;
    })
}


function getTopClasses(mat, labels, k = 5) {
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
    for (let i = 0; i < k; ++i) {
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

function printResult(classes, topNum){
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
