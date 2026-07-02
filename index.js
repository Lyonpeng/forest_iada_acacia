const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const cors = require("cors");
const archiverPkg = require("archiver");
const archiver = archiverPkg.default || archiverPkg;
const app = express();

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));   // parse POSTed form bodies
app.use(express.static("public"));

/* -------------------------
   OPEN RESULT VIA POST
   batchId comes in the request body (not the URL) and is injected
   into the served result.html as window.BATCH_ID.
-------------------------- */
app.post("/result", (req, res) => {

    const batchId = req.body.batchId;
    if (!batchId) {
        return res.status(400).send("Missing batchId");
    }

    const templatePath = path.join(__dirname, "public", "result.html");
    let html = fs.readFileSync(templatePath, "utf-8");

    // JSON.stringify safely escapes quotes so it can't break the script/HTML
    const inject = `<script>window.BATCH_ID = ${JSON.stringify(String(batchId))};</script>`;
    html = html.replace("</head>", `${inject}\n</head>`);

    res.send(html);
});

const UPLOAD_DIR = path.join(__dirname, "uploads");

if (!fs.existsSync(UPLOAD_DIR)) {
    fs.mkdirSync(UPLOAD_DIR);
}

/* -------------------------
   MULTER STORAGE
-------------------------- */
const storage = multer.diskStorage({

    destination: (req, file, cb) => {

        const folderName = req.body.batchId;

        if (!folderName) {
            return cb(new Error("Missing batchId"));
        }

        const uploadPath = path.join(
            UPLOAD_DIR,
            folderName
        );

        fs.mkdirSync(uploadPath, {
            recursive: true
        });

        cb(null, uploadPath);
    },

    filename: (req, file, cb) => {
        cb(null, file.originalname);
    }
});
const upload = multer({ storage });
const { spawn } = require("child_process");

app.post("/upload", upload.array("file"), (req, res) => {

    console.log("FILES RECEIVED:", req.files);

    if (!req.files || req.files.length === 0) {
        return res.status(400).json({
            success: false,
            message: "No files received"
        });
    }

    res.json({
        success: true,
        count: req.files.length,
        files: req.files
    });
});

app.post("/process", (req, res) => {

    console.log("Running Python inference...");
    const BatchId = req.body.batchId;
    const PYTHON_BIN = process.env.PYTHON_BIN || "python";
    const python = spawn(PYTHON_BIN, [
        "-u",                       // unbuffered: stream Python output in real time
        "python/detection.py",
        BatchId
    ], {
        // pipe the child's stdout/stderr straight to this terminal
        stdio: ["ignore", "inherit", "inherit"],
        env: { ...process.env, PYTHONUNBUFFERED: "1" }
    });

    python.on("close", (code) => {
        console.log("Python finished with code:", code);

        res.json({
            batchId: BatchId,
            status: "completed",
        });
    });
});

app.use("/uploads", express.static(UPLOAD_DIR));

/* -------------------------
   LIST VISUAL IMAGES FOR A BATCH
-------------------------- */
app.get("/images/:batchId", (req, res) => {

    const batchId = req.params.batchId;
    const visualDir = path.join(UPLOAD_DIR, batchId, "visuals");

    if (!fs.existsSync(visualDir)) {
        return res.json([]);
    }

    const files = fs.readdirSync(visualDir)
        .filter(name => /\.(png|jpe?g|gif|webp)$/i.test(name))
        .map(name => `/uploads/${batchId}/visuals/${name}`);

    res.json(files);
});
app.get("/download/:batchId", (req, res) => {

    const folderPath = path.join(__dirname, "uploads", req.params.batchId);

    if (!fs.existsSync(folderPath)) {
        return res.status(404).send("Folder not found");
    }

    res.attachment(`${req.params.batchId}.zip`);

    const archive = archiver("zip", {
        zlib: { level: 9 }
    });

    archive.on("error", (err) => {
        console.error("ZIP error:", err);
        res.status(500).send("Zip failed");
    });

    archive.pipe(res);
    archive.directory(folderPath, false);
    archive.finalize();
});
/* -------------------------
   DELETE A BATCH FOLDER
-------------------------- */
app.delete("/batches/:batchId", (req, res) => {

    const batchId = req.params.batchId;
    const folderPath = path.join(UPLOAD_DIR, batchId);

    // guard against path traversal (e.g. "../something")
    const resolved = path.resolve(folderPath);
    if (resolved !== path.resolve(UPLOAD_DIR, batchId) ||
        path.dirname(resolved) !== path.resolve(UPLOAD_DIR)) {
        return res.status(400).json({ success: false, message: "Invalid batchId" });
    }

    if (!fs.existsSync(resolved)) {
        return res.status(404).json({ success: false, message: "Folder not found" });
    }

    try {
        fs.rmSync(resolved, { recursive: true, force: true });
        console.log("Deleted batch folder:", resolved);
        res.json({ success: true, batchId });
    } catch (err) {
        console.error("Delete failed:", err);
        res.status(500).json({ success: false, message: err.message });
    }
});

app.get("/batches", (req, res) => {

    const uploadsDir = UPLOAD_DIR;

    const folders = fs.readdirSync(uploadsDir);

    const results = folders.map(folder => {

        const statusPath = path.join(uploadsDir, folder, "status.json");

        let statusData = {
            batchId: folder,
            status: "unknown",
            progress: 0
        };

        if (fs.existsSync(statusPath)) {
            try {
                const raw = fs.readFileSync(statusPath, "utf-8");
                statusData = JSON.parse(raw);
            } catch (e) {
                console.log("Error reading status:", folder);
            }
        }

        return statusData;
    });

    res.json(results);
});
/* -------------------------
   START SERVER
-------------------------- */
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});