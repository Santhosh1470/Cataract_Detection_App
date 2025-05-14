document.getElementById("uploadForm").addEventListener("submit", function(event) {
    event.preventDefault();
    
    document.getElementById("loading").style.display = "block";

    let formData = new FormData(this);
    fetch("/detect", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("loading").style.display = "none";
        document.getElementById("result").style.display = "block";
        
        document.getElementById("uploadedImage").src = data.image_url;
        document.getElementById("predictionText").innerText = `Prediction: ${data.prediction}`;
    })
    .catch(error => console.error("Error:", error));
});

function goBack() {
    document.getElementById("result").style.display = "none";
    document.getElementById("uploadForm").reset();
}
