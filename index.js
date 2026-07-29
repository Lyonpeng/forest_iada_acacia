const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const os = require("os");
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
   STATUS.JSON HELPERS
   Node-side failures during /upload and /process are recorded here so
   /batches surfaces them. (Python records its own errors via tracker().)
-------------------------- */
function statusPathFor(batchId) {
    return path.join(UPLOAD_DIR, batchId, "status.json");
}

// human-readable timestamp, e.g. "23/7/2026 10:39 AM" (D/M/YYYY h:mm AM/PM)
function formatTime(date) {
    const day = date.getDate();
    const month = date.getMonth() + 1;
    const year = date.getFullYear();
    const minutes = String(date.getMinutes()).padStart(2, "0");
    const ampm = date.getHours() >= 12 ? "PM" : "AM";
    let hours = date.getHours() % 12;
    if (hours === 0) hours = 12;
    return `${day}/${month}/${year} ${hours}:${minutes} ${ampm}`;
}

// current status.json for a batch, or a sane default if missing/corrupt
function readStatus(batchId) {
    const statusPath = statusPathFor(batchId);

    if (fs.existsSync(statusPath)) {
        try {
            return JSON.parse(fs.readFileSync(statusPath, "utf-8"));
        } catch (e) {
            console.log("Error reading status:", batchId);
        }
    }

    return {
        batchId,
        status: "unknown",
        time: "",
        total: "",
        processed: "",
        progress: 0,
        error: ""
    };
}

// stamp a Node-side failure into status.json (keeps the existing fields)
function writeError(batchId, message) {
    if (!batchId) {
        return;
    }

    try {
        fs.mkdirSync(path.join(UPLOAD_DIR, batchId), { recursive: true });

        const data = readStatus(batchId);
        data.batchId = batchId;
        data.status = "error";
        data.error = String(message);

        fs.writeFileSync(statusPathFor(batchId), JSON.stringify(data, null, 4));
    } catch (e) {
        console.error("Failed to write error status for", batchId, e);
    }
}

// flag a batch as interrupted, preserving its other fields (time, progress…)
function markInterrupted(batchId) {
    if (!batchId) {
        return;
    }

    try {
        fs.mkdirSync(path.join(UPLOAD_DIR, batchId), { recursive: true });

        const data = readStatus(batchId);
        data.batchId = batchId;
        data.status = "interrupted";

        fs.writeFileSync(statusPathFor(batchId), JSON.stringify(data, null, 4));
    } catch (e) {
        console.error("Failed to mark interrupted for", batchId, e);
    }
}

// batchId -> running Python child process, so /interrupt can stop it
const runningProcesses = {};

// pull the exception line out of a Python traceback,
// e.g. "ModuleNotFoundError: No module named 'torch'"
function extractPythonError(stderr) {
    const lines = (stderr || "")
        .split(/\r?\n/)
        .map(l => l.trim())
        .filter(Boolean);

    if (lines.length === 0) {
        return "";
    }

    // scan up from the bottom for the "SomethingError/Exception: ..." line
    for (let i = lines.length - 1; i >= 0; i--) {
        if (/^[\w.]+(Error|Exception):/.test(lines[i])) {
            return lines[i];
        }
    }

    // no recognisable exception line — fall back to the last line of output
    return lines[lines.length - 1];
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

        // returns the created path only when the folder is newly made,
        // undefined if it already existed
        const created = fs.mkdirSync(uploadPath, {
            recursive: true
        });

        // seed status.json once, on first creation of the batch folder
        if (created) {
            const statusPath = path.join(uploadPath, "status.json");
            fs.writeFileSync(statusPath, JSON.stringify({
                batchId: folderName,
                status: "initiating",
                time: formatTime(new Date()),
                total: "",
                processed: "",
                progress: 0,
                error: ""
            }, null, 4));
        }

        cb(null, uploadPath);
    },

    filename: (req, file, cb) => {
        cb(null, file.originalname);
    }
});
app.get("/validate-batch/:batchId", (req, res) => {
    const batchId = req.params.batchId.trim();
    const batchPath = path.join(UPLOAD_DIR, batchId);

    res.json({
        exists: fs.existsSync(batchPath)
    });
});
const upload = multer({ storage });
const { spawn } = require("child_process");

/* -------------------------
   MULTER STORAGE FOR EXIF VALIDATION
   Images are dropped into a throwaway OS temp folder (one per request) just
   long enough for validate_metadata.py to inspect them; it's deleted after.
-------------------------- */
const validateStorage = multer.diskStorage({

    destination: (req, file, cb) => {
        if (!req.validateTempDir) {
            req.validateTempDir = fs.mkdtempSync(
                path.join(os.tmpdir(), "exif-validate-")
            );
        }
        cb(null, req.validateTempDir);
    },

    filename: (req, file, cb) => {
        cb(null, file.originalname);
    }
});
const validateUpload = multer({ storage: validateStorage }).array("file");

const uploadArray = upload.array("file");

app.post("/upload", (req, res) => {

    // run multer manually so storage/disk errors don't hit the default
    // HTML error page — record them in status.json instead
    uploadArray(req, res, (err) => {

        const batchId = req.body && req.body.batchId;

        if (err) {
            console.error("Upload error:", err);
            writeError(batchId, err.message || String(err));
            return res.status(500).json({
                success: false,
                message: err.message || "Upload failed"
            });
        }

        console.log("FILES RECEIVED:", req.files);

        if (!req.files || req.files.length === 0) {
            writeError(batchId, "No files received");
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
});

app.post("/process", (req, res) => {

    console.log("Running Python inference...");
    const BatchId = req.body.batchId;

    if (!BatchId) {
        return res.status(400).json({
            batchId: "",
            status: "error",
            error: "Missing batchId"
        });
    }

    // customisation from the front-end set-up
    const parsedWidth = parseFloat(req.body.sensorWidth);
    const sensorWidth = Number.isFinite(parsedWidth) && parsedWidth > 0
        ? String(parsedWidth)
        : "17.3";   // default: DJI Zenmuse L2 (17.3 mm)

    const PYTHON_BIN = process.env.PYTHON_BIN || "python";

    let python;
    try {
        python = spawn(PYTHON_BIN, [
            "-u",                       // unbuffered: stream Python output in real time
            "python/detection.py",
            BatchId,
            sensorWidth
        ], {
            // stdout -> terminal; stderr piped so we can capture the traceback
            stdio: ["ignore", "inherit", "pipe"],
            env: { ...process.env, PYTHONUNBUFFERED: "1" }
        });
    } catch (err) {
        // spawn threw synchronously (bad args, etc.)
        console.error("Failed to start Python:", err);
        writeError(BatchId, `Failed to start Python: ${err.message}`);
        return res.status(500).json({
            batchId: BatchId,
            status: "error",
            error: err.message
        });
    }

    // track the live process so /interrupt can stop it
    runningProcesses[BatchId] = python;

    // capture Python's stderr (still echo it to the terminal). keep only the
    // tail so a runaway traceback can't grow unbounded — the exception line
    // we want sits at the very end anyway.
    let stderrBuf = "";
    if (python.stderr) {
        python.stderr.on("data", (chunk) => {
            const text = chunk.toString();
            process.stderr.write(text);
            stderrBuf = (stderrBuf + text).slice(-8192);
        });
    }

    // guard so "error" + "close" can't both send a response
    let responded = false;

    // fires when the process could not be spawned (e.g. python not found)
    python.on("error", (err) => {
        console.error("Failed to start Python:", err);
        delete runningProcesses[BatchId];
        if (responded) return;
        responded = true;

        writeError(BatchId, `Failed to start Python: ${err.message}`);
        res.status(500).json({
            batchId: BatchId,
            status: "error",
            error: err.message
        });
    });

    python.on("close", (code) => {
        console.log("Python finished with code:", code);
        delete runningProcesses[BatchId];
        if (responded) return;
        responded = true;

        // interrupted by the user: /interrupt already wrote "interrupted" and
        // killed the process — leave that status as-is, don't record an error
        if (readStatus(BatchId).status === "interrupted") {
            return res.json({ batchId: BatchId, status: "interrupted" });
        }

        if (code !== 0) {
            // prefer the real Python exception (e.g. "ModuleNotFoundError: ...")
            // over a generic exit-code message
            const message = extractPythonError(stderrBuf)
                || `Python exited with code ${code}`;

            // Python normally records its own error in status.json; only stamp
            // one here if it exited abnormally without doing so.
            if (readStatus(BatchId).status !== "error") {
                writeError(BatchId, message);
            }

            return res.status(500).json({
                batchId: BatchId,
                status: "error",
                error: message
            });
        }

        res.json({
            batchId: BatchId,
            status: "completed",
        });
    });
});

/* -------------------------
   INTERRUPT A RUNNING BATCH
   Kills the batch's Python process and marks status.json "interrupted".
-------------------------- */
app.post("/interrupt/:batchId", (req, res) => {
    console.log("Interrupt request for batch:", req.params.batchId);
    const batchId = req.params.batchId;
    const folderPath = path.join(UPLOAD_DIR, batchId);

    // guard against path traversal (e.g. "../something")
    const resolved = path.resolve(folderPath);
    if (resolved !== path.resolve(UPLOAD_DIR, batchId) ||
        path.dirname(resolved) !== path.resolve(UPLOAD_DIR)) {
        return res.status(400).json({ success: false, message: "Invalid batchId" });
    }

    const child = runningProcesses[batchId];
    const status = readStatus(batchId);
    const statusRunning = status.status === "initiating" || status.status === "processing";
    const isComplete = status.progress == 100;

    // interruptible if we still hold a live process, or the recorded status is
    // running and not yet complete (a finished batch keeps status "processing")
    if (!child && (!statusRunning || isComplete)) {
        return res.status(409).json({ success: false, message: "Batch is not running" });
    }

    // record "interrupted" first (preserves time/progress) so the process's own
    // close handler sees it and doesn't overwrite it with an error
    markInterrupted(batchId);

    if (child) {
        try {
            child.kill();   // SIGTERM -> TerminateProcess on Windows
        } catch (e) {
            console.error("Failed to kill process for", batchId, e);
        }
        delete runningProcesses[batchId];
    }

    console.log("Interrupted batch:", batchId);
    res.json({ success: true, batchId, status: "interrupted" });
});

/* -------------------------
   VALIDATE EXIF METADATA (pre-upload)
   Stores the dropped images in a throwaway temp folder, runs
   validate_metadata.py (exiftool) against it, then deletes the folder.
-------------------------- */
app.post("/validate", (req, res) => {

    // run multer manually so we can clean up the temp folder on any error
    validateUpload(req, res, (err) => {
        const tempDir = req.validateTempDir;
        const cleanup = () => {
            if (tempDir) {
                fs.rm(tempDir, { recursive: true, force: true }, (rmErr) => {
                    if (rmErr) {
                        console.error("Failed to remove temp dir:", tempDir, rmErr);
                    }
                });
            }
        };

        if (err) {
            console.error("Validate upload error:", err);
            cleanup();
            return res.status(500).json({
                ok: false,
                message: err.message || "Upload failed"
            });
        }

        if (!req.files || req.files.length === 0) {
            cleanup();
            return res.status(400).json({
                ok: false,
                message: "No files received"
            });
        }

        const PYTHON_BIN = process.env.PYTHON_BIN || "python";

        let python;
        try {
            python = spawn(PYTHON_BIN, [
                "-u",
                "python/validate_metadata.py",
                tempDir
            ], {
                stdio: ["ignore", "pipe", "pipe"],
                env: { ...process.env, PYTHONUNBUFFERED: "1" }
            });
        } catch (spawnErr) {
            console.error("Failed to start validator:", spawnErr);
            cleanup();
            return res.status(500).json({
                ok: false,
                message: `Failed to start validator: ${spawnErr.message}`
            });
        }

        let stdout = "";
        let stderr = "";
        python.stdout.on("data", (chunk) => { stdout += chunk.toString(); });
        python.stderr.on("data", (chunk) => {
            const text = chunk.toString();
            stderr = (stderr + text).slice(-8192);
            process.stderr.write(text);
        });

        // guard so "error" + "close" can't both send a response
        let responded = false;

        python.on("error", (spawnErr) => {
            console.error("Failed to start validator:", spawnErr);
            if (responded) return;
            responded = true;
            cleanup();
            res.status(500).json({
                ok: false,
                message: `Failed to start validator: ${spawnErr.message}`
            });
        });

        python.on("close", (code) => {
            if (responded) return;
            responded = true;
            cleanup();   // images have been inspected — safe to delete now

            if (code !== 0) {
                return res.status(500).json({
                    ok: false,
                    message: extractPythonError(stderr)
                        || `Validator exited with code ${code}`
                });
            }

            try {
                return res.json(JSON.parse(stdout));
            } catch (parseErr) {
                console.error("Bad validator output:", stdout);
                return res.status(500).json({
                    ok: false,
                    message: "Could not parse validator output"
                });
            }
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

    // if it's somehow still running, stop the process before removing its files
    const child = runningProcesses[batchId];
    if (child) {
        try { child.kill(); } catch (e) { /* ignore */ }
        delete runningProcesses[batchId];
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
            time: "",
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