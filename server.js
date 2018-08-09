const express= require("express");

let app= express();

app.get("/", (req, res) => {
    console.log("\n\n\tnew Index Request--");
    res.sendFile(__dirname+ "/public/index.html");
});

app.use((req, res, next) => {
    console.log("  Requested- "+ new Date().toTimeString());
    console.log("\t sending- "+req.url);
    next();
});

app.use(express.static(__dirname+"/public"));


app.listen(3000);