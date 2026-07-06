// virtualpainter.js
// Uses MediaPipe Hands to track the index finger over the webcam
// video feed and draws strokes onto the existing #drawingCanvas.
// Relies on global classes (Hands, Camera) loaded via CDN <script> tags.
//
// This revision fixes the coordinate-mapping bugs (mirroring / offset),
// adds a debug landmark overlay + live debug panel (created dynamically
// in JS so no HTML changes are required), adds gesture hysteresis, and
// adds exponential smoothing for the pen line. See the long comment
// blocks near each function for the "why".

// ---- Config ----
const DRAW_COLOR = "#F891A5";
const LINE_WIDTH = 15;
const ERASER_RADIUS = 50;
const HYSTERESIS_FRAMES = 4;
const SMOOTHING_ALPHA = 0.5;
const MIRROR_CAMERA = true; // single source of truth for video, landmarks, drawing, and upload orientation

// ---- Core state ----
let videoEl = null;
let canvasEl = null;
let ctx = null;
let prevX = null;
let prevY = null;
let handsInstance = null;
let cameraInstance = null;

// FEATURE: pause/resume state
let isCameraPaused = false;

// Debug-mode tracking (only log when it actually changes)
let currentMode = null; // "DRAW" | "ERASE" | "IDLE"

// ---- Gesture hysteresis state ----
let stableMode = "IDLE";
let pendingMode = null;
let pendingModeCount = 0;

// ---- Smoothing state (for the drawn point, separate from raw fingertip) ----
let smoothX = null;
let smoothY = null;

// ---- Coordinate-mapping cache (recomputed on resize / metadata load) ----
// NOTE: these must stay as plain `let` declarations initialized to safe
// defaults. They were previously (accidentally) turned into top-level
// function calls (`displayGeometry = computeVideoDisplayGeometry();`),
// which executed immediately at module-load time — before `videoEl` is
// ever assigned in initPainter(). That call chain does
// `videoEl.getBoundingClientRect()` on a `null` videoEl and throws,
// breaking the whole script before initPainter() ever runs. The real
// values are computed later, inside resizeCanvasToVideo(), once videoEl
// exists.
let displayGeometry = null; // see computeVideoDisplayGeometry()
let isMirrored = false;     // derived from MIRROR_CAMERA in resizeCanvasToVideo()

// ---- Debug overlay elements (created dynamically, no HTML edits needed) ----
let landmarkCanvasEl = null;
let landmarkCtx = null;
let debugPanelEl = null;

// ---- FPS tracking ----
let lastFrameTs = null;
let fps = 0;

// Standard 21-point MediaPipe hand skeleton connections, used only for
// the debug overlay (not the drawing canvas).
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],       // thumb
    [0, 5], [5, 6], [6, 7], [7, 8],       // index
    [5, 9], [9, 10], [10, 11], [11, 12],  // middle
    [9, 13], [13, 14], [14, 15], [15, 16],// ring
    [13, 17], [17, 18], [18, 19], [19, 20], // pinky
    [0, 17],                              // palm base
];

/* =========================================================================
 * TASK 1/2/8 — COORDINATE MAPPING
 * ========================================================================= */

/**
 * Detects whether the *displayed* <video> is mirrored via CSS
 * (e.g. `transform: scaleX(-1)`). No longer used to drive `isMirrored`
 * (that now comes directly from MIRROR_CAMERA, see resizeCanvasToVideo),
 * but kept available as a diagnostic/manual utility.
 */
function detectMirrored() {
    const transform = getComputedStyle(videoEl).transform;
    if (!transform || transform === "none") return false;

    const match = transform.match(/matrix\(([^)]+)\)/);
    if (!match) return false;

    const values = match[1].split(",").map((v) => parseFloat(v.trim()));
    const a = values[0]; // matrix(a, b, c, d, e, f) — 'a' scales the x-axis
    return a < 0;
}

/**
 * Diagnostic-only check: warns in the console if the *drawing canvas
 * itself* also carries a horizontal-flip transform.
 *
 * BUG CLASS: "double mirroring" happens when BOTH the video and the
 * canvas layered on top of it are CSS-mirrored. In that case our
 * coordinate math correctly un-mirrors the landmark once, but the
 * browser then mirrors the whole canvas element a second time when
 * painting it to the screen, putting the stroke back on the wrong
 * side. This can't be fixed from JS (it's a CSS layering issue), so
 * we surface it loudly instead of guessing.
 */
function warnIfCanvasMirrored() {
    const transform = getComputedStyle(canvasEl).transform;
    if (!transform || transform === "none") return;
    const match = transform.match(/matrix\(([^)]+)\)/);
    if (match && parseFloat(match[1].split(",")[0]) < 0) {
        console.warn(
            "[virtualpainter] #drawingCanvas has a horizontal-flip CSS transform. " +
            "Only the <video> should be mirrored, not the canvas layered on top of it — " +
            "having both will double-mirror every stroke. Remove the transform from the canvas."
        );
    }
}

/**
 * Computes how the video's native frame maps onto its displayed CSS box,
 * accounting for `object-fit` (fill / cover / contain).
 *
 * BUG FOUND (Task 1/8 — "CSS scaling" / "canvas/video size mismatch"):
 * MediaPipe's Camera utility grabs the *entire native* video frame
 * (videoWidth x videoHeight) and reports landmarks normalized 0-1 over
 * that full frame — regardless of how the video is displayed on screen.
 * The previous code assumed a naive 1:1 stretch (`landmark.x * canvas.width`),
 * which is only correct when `object-fit: fill` (the default) is used and
 * the canvas exactly overlays the video's box. If `object-fit: cover` is
 * used anywhere (very common for full-bleed webcam backgrounds), part of
 * the native frame is cropped from view, and a naive stretch mapping
 * increasingly drifts from the real fingertip position toward the edges
 * — matching your "sometimes draws away from where my finger actually
 * is" symptom. `object-fit: contain` has the opposite problem (letterbox
 * bars) which also throws off a naive mapping.
 *
 * FIX: read the actual native video resolution + the computed
 * `object-fit`, and compute the crop/letterbox so normalized landmark
 * coordinates map to the exact same pixel the eye sees, for all three
 * object-fit modes. For the common `fill` case this reduces to exactly
 * the original simple math, so nothing changes for that setup.
 */
function computeVideoDisplayGeometry() {
    const rect = videoEl.getBoundingClientRect();
    const vw = videoEl.videoWidth || rect.width;
    const vh = videoEl.videoHeight || rect.height;
    const objectFit = getComputedStyle(videoEl).objectFit || "fill";

    const geometry = {
        rect,
        objectFit,
        // For "cover": fraction of the native frame that's actually visible,
        // and the normalized offset into the native frame where it starts.
        visibleFracW: 1,
        visibleFracH: 1,
        offsetFracX: 0,
        offsetFracY: 0,
        // For "contain": the letterboxed drawn area within rect (in px).
        letterboxX: 0,
        letterboxY: 0,
        drawW: rect.width,
        drawH: rect.height,
    };

    const containerAspect = rect.width / rect.height;
    const videoAspect = vw / vh;

    if (objectFit === "cover") {
        if (videoAspect > containerAspect) {
            geometry.visibleFracW = containerAspect / videoAspect;
            geometry.offsetFracX = (1 - geometry.visibleFracW) / 2;
        } else {
            geometry.visibleFracH = videoAspect / containerAspect;
            geometry.offsetFracY = (1 - geometry.visibleFracH) / 2;
        }
    } else if (objectFit === "contain") {
        if (videoAspect > containerAspect) {
            geometry.drawW = rect.width;
            geometry.drawH = rect.width / videoAspect;
            geometry.letterboxY = (rect.height - geometry.drawH) / 2;
        } else {
            geometry.drawH = rect.height;
            geometry.drawW = rect.height * videoAspect;
            geometry.letterboxX = (rect.width - geometry.drawW) / 2;
        }
    }
    // objectFit === "fill" (default): geometry stays at its identity values.

    return geometry;
}

/**
 * Converts a normalized MediaPipe landmark (0-1 range, relative to the
 * FULL native camera frame) into actual canvas pixel coordinates,
 * accounting for mirroring and object-fit cropping/letterboxing.
 *
 * Returns { x, y, visible }. `visible` is false when the point falls
 * outside the on-screen crop (only possible with object-fit: cover) —
 * callers should treat that like "fingertip not currently on screen".
 */
function landmarkToCanvasCoords(landmark) {
    const g = displayGeometry;
    let nx = landmark.x;
    let ny = landmark.y;
    let visible = true;

    if (g.objectFit === "cover") {
        // Re-normalize from "fraction of full frame" to "fraction of visible crop"
        nx = (nx - g.offsetFracX) / g.visibleFracW;
        ny = (ny - g.offsetFracY) / g.visibleFracH;
        if (nx < 0 || nx > 1 || ny < 0 || ny > 1) visible = false;
        if (isMirrored) nx = 1 - nx;
        return { x: nx * canvasEl.width, y: ny * canvasEl.height, visible };
    }

    if (g.objectFit === "contain") {
        let px = nx * g.drawW + g.letterboxX;
        let py = ny * g.drawH + g.letterboxY;
        if (isMirrored) px = g.rect.width - px; // letterbox is symmetric, safe to mirror around full rect
        // Canvas is sized to rect, so px/py (in rect-space) map 1:1 to canvas pixel space.
        return { x: px, y: py, visible: true };
    }

    // "fill" (default): direct stretch — identical to the original simple math,
    // just with mirroring now correctly conditional instead of hardcoded.
    if (isMirrored) nx = 1 - nx;
    return { x: nx * canvasEl.width, y: ny * canvasEl.height, visible: true };
}

/**
 * Resize the drawing canvas (and the debug landmark canvas) to match
 * the video element's displayed size, and refresh the cached mirroring
 * / object-fit geometry so coordinate mapping stays correct after any
 * layout change (window resize, orientation change, etc).
 */
function resizeCanvasToVideo() {
    const rect = videoEl.getBoundingClientRect();
    canvasEl.width = rect.width;
    canvasEl.height = rect.height;

    if (landmarkCanvasEl) {
        landmarkCanvasEl.width = rect.width;
        landmarkCanvasEl.height = rect.height;
        landmarkCanvasEl.style.top = canvasEl.offsetTop + "px";
        landmarkCanvasEl.style.left = canvasEl.offsetLeft + "px";
        landmarkCanvasEl.style.width = rect.width + "px";
        landmarkCanvasEl.style.height = rect.height + "px";
    }

    displayGeometry = computeVideoDisplayGeometry();
    // Mirroring is driven directly by the MIRROR_CAMERA config constant
    // (not sniffed from CSS), so toggling MIRROR_CAMERA is the one
    // switch that updates video + landmarks + drawing + upload together.
    isMirrored = MIRROR_CAMERA;
}

/**
 * Handles the case where the video's metadata was already available by
 * the time this script ran (so the "loadedmetadata" listener would never
 * fire). Previously this left the canvas at its default 300x150 size,
 * causing strokes to appear misaligned/missing.
 */
function ensureCanvasSized() {
    if (videoEl.readyState >= 1 /* HAVE_METADATA */) {
        resizeCanvasToVideo();
    }
}

/* =========================================================================
 * TASK 3 — LANDMARK DEBUG OVERLAY (separate transparent canvas)
 * ========================================================================= */

/**
 * Creates a transparent overlay canvas ABOVE the drawing canvas, purely
 * via JS (no HTML edits). It borrows #drawingCanvas's own position/size
 * so it lines up exactly, and sits one z-index above it.
 *
 * Layer order (top -> bottom): landmarkCanvas > drawingCanvas > video.
 *
 * NOTE: this assumes #drawingCanvas's parent element is a positioning
 * context (e.g. `position: relative`) for absolutely-positioned children,
 * which must already be true today for the drawing canvas to overlay the
 * video correctly. If that's ever not the case, add
 * `position: relative` to the shared parent — that's the only CSS change
 * this feature could require, and only if your layout doesn't already do it.
 */
function createLandmarkCanvas() {
    if (landmarkCanvasEl) return;

    landmarkCanvasEl = document.createElement("canvas");
    landmarkCanvasEl.id = "landmarkCanvas";

    const canvasStyle = getComputedStyle(canvasEl);
    Object.assign(landmarkCanvasEl.style, {
        position: canvasStyle.position === "static" ? "absolute" : canvasStyle.position,
        top: canvasEl.offsetTop + "px",
        left: canvasEl.offsetLeft + "px",
        width: canvasStyle.width,
        height: canvasStyle.height,
        pointerEvents: "none", // never intercept clicks/gestures meant for the page
        zIndex: String((parseInt(canvasStyle.zIndex, 10) || 0) + 1),
    });

    canvasEl.parentElement.appendChild(landmarkCanvasEl);
    landmarkCtx = landmarkCanvasEl.getContext("2d");
}

/**
 * Clears and redraws all 21 landmarks + skeleton connections on the
 * debug overlay canvas (never on the drawing canvas).
 */
function drawLandmarksOverlay(landmarks) {
    if (!landmarkCtx) return;
    landmarkCtx.clearRect(0, 0, landmarkCanvasEl.width, landmarkCanvasEl.height);

    // Map every landmark once so we don't repeat the mapping math per line.
    const points = landmarks.map((lm) => landmarkToCanvasCoords(lm));

    // Skeleton connections
    landmarkCtx.strokeStyle = "rgba(255, 255, 255, 0.8)";
    landmarkCtx.lineWidth = 2;
    for (const [a, b] of HAND_CONNECTIONS) {
        landmarkCtx.beginPath();
        landmarkCtx.moveTo(points[a].x, points[a].y);
        landmarkCtx.lineTo(points[b].x, points[b].y);
        landmarkCtx.stroke();
    }

    // All 21 landmark dots
    landmarkCtx.fillStyle = "#00BFFF";
    for (const p of points) {
        landmarkCtx.beginPath();
        landmarkCtx.arc(p.x, p.y, 3, 0, Math.PI * 2);
        landmarkCtx.fill();
    }

    // Highlighted tips, each a different color
    const highlight = (index, color, radius) => {
        const p = points[index];
        landmarkCtx.fillStyle = color;
        landmarkCtx.beginPath();
        landmarkCtx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        landmarkCtx.fill();
    };
    highlight(4, "#FF3B30", 7);  // thumb tip - red
    highlight(8, "#00FF00", 7); // index tip - green (matches pen color)
    highlight(12, "#FFD60A", 7); // middle tip - yellow
}

/* =========================================================================
 * TASK 4 — LIVE DEBUG PANEL
 * ========================================================================= */

/**
 * Creates a small on-screen debug panel via JS (no HTML edits needed).
 */
function createDebugPanel() {
    if (debugPanelEl) return;

    debugPanelEl = document.createElement("div");
    debugPanelEl.id = "debugPanel";
    Object.assign(debugPanelEl.style, {
        position: "absolute",
        top: "8px",
        left: "8px",
        padding: "8px 10px",
        background: "rgba(0, 0, 0, 0.65)",
        color: "#0f0",
        fontFamily: "monospace",
        fontSize: "12px",
        lineHeight: "1.4",
        whiteSpace: "pre",
        borderRadius: "6px",
        pointerEvents: "none",
        zIndex: String((parseInt(getComputedStyle(canvasEl).zIndex, 10) || 0) + 2),
    });

    canvasEl.parentElement.appendChild(debugPanelEl);
}

function updateDebugPanel({ mode, rawX, rawY, canvasX, canvasY, confidence }) {
    if (!debugPanelEl) return;

    const fmt = (n) => (typeof n === "number" ? n.toFixed(3) : "N/A");
    const fmtInt = (n) => (typeof n === "number" ? Math.round(n) : "N/A");

    debugPanelEl.textContent =
        `Mode: ${mode}\n` +
        `Index (norm): x=${fmt(rawX)} y=${fmt(rawY)}\n` +
        `Index (canvas px): x=${fmtInt(canvasX)} y=${fmtInt(canvasY)}\n` +
        `Canvas size: ${canvasEl.width} x ${canvasEl.height}\n` +
        `Video size: ${videoEl.videoWidth} x ${videoEl.videoHeight}\n` +
        `Mirrored: ${isMirrored}\n` +
        `FPS: ${fps.toFixed(1)}\n` +
        `Confidence: ${confidence !== null ? confidence.toFixed(2) : "N/A"}`;
}

function updateFps() {
    const now = performance.now();
    if (lastFrameTs !== null) {
        const delta = now - lastFrameTs;
        const instantFps = delta > 0 ? 1000 / delta : 0;
        fps = fps === 0 ? instantFps : fps * 0.9 + instantFps * 0.1; // light smoothing so it doesn't jitter
    }
    lastFrameTs = now;
}

/* =========================================================================
 * DRAWING / ERASING
 * ========================================================================= */

/**
 * Draws a line segment from the previous (smoothed) point to the current
 * (smoothed) point on the drawing canvas.
 */
function drawLine(x, y) {
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = DRAW_COLOR;
    ctx.lineWidth = LINE_WIDTH;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    if (prevX === null || prevY === null) {
        // First point of a new stroke: draw a dot so a single tap still shows something.
        ctx.beginPath();
        ctx.arc(x, y, LINE_WIDTH / 2, 0, Math.PI * 2);
        ctx.fillStyle = DRAW_COLOR;
        ctx.fill();
    } else {
        ctx.beginPath();
        ctx.moveTo(prevX, prevY);
        ctx.lineTo(x, y);
        ctx.stroke();
    }

    prevX = x;
    prevY = y;
}

/**
 * Erases a circular area around the fingertip instead of clearing the
 * whole canvas. Uses destination-out so only pixels under the circle
 * become transparent, then restores source-over for subsequent draws.
 */
function eraseAt(x, y) {
    ctx.save();
    ctx.globalCompositeOperation = "destination-out";
    ctx.beginPath();
    ctx.arc(x, y, ERASER_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore(); // restores globalCompositeOperation back to "source-over"
}

/**
 * Resets stroke continuity AND the smoothing filter together, so a new
 * gesture never jump-connects to (or gets dragged by) a previous one.
 */
function resetStroke() {
    prevX = null;
    prevY = null;
    smoothX = null;
    smoothY = null;
}

/**
 * TASK 6 — exponential smoothing of the fingertip point before it's used
 * for drawing/erasing, so the line feels like a pen instead of a jittery
 * scatter of noisy MediaPipe points.
 */
function smoothPoint(x, y) {
    if (smoothX === null || smoothY === null) {
        smoothX = x;
        smoothY = y;
    } else {
        smoothX = SMOOTHING_ALPHA * x + (1 - SMOOTHING_ALPHA) * smoothX;
        smoothY = SMOOTHING_ALPHA * y + (1 - SMOOTHING_ALPHA) * smoothY;
    }
    return { x: smoothX, y: smoothY };
}

/**
 * Clears the entire drawing canvas. Exposed for optional manual use.
 */
function clearCanvas() {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    resetStroke();
}

function setMode(mode) {
    if (mode !== currentMode) {
        currentMode = mode;
        console.log("Current mode:", mode);
    }
}

/* =========================================================================
 * TASK 5 — GESTURE DETECTION + HYSTERESIS
 * ========================================================================= */

/**
 * Returns [thumb, index, middle, ring, pinky] booleans.
 *
 * Non-thumb fingers use tip.y vs pip.y, which is mirror-safe (mirroring
 * is a horizontal flip and doesn't affect the y axis).
 *
 * Thumb uses a distance-from-wrist comparison instead of an x-position
 * comparison, so it doesn't depend on hand orientation or mirroring.
 */
function getFingersUp(landmarks) {
    const fingersUp = [false, false, false, false, false];
    const wrist = landmarks[0];
    const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

    const thumbTipDist = dist(landmarks[4], wrist);
    const thumbMcpDist = dist(landmarks[2], wrist);
    fingersUp[0] = thumbTipDist > thumbMcpDist * 1.2;

    const tips = [8, 12, 16, 20];
    const pips = [6, 10, 14, 18];
    for (let i = 0; i < tips.length; i++) {
        fingersUp[i + 1] = landmarks[tips[i]].y < landmarks[pips[i]].y;
    }

    return fingersUp;
}

/**
 * Debounces the raw per-frame gesture into a "stable" mode, requiring
 * HYSTERESIS_FRAMES consecutive identical readings before switching.
 * This is what stops DRAW/ERASE/IDLE from flickering when MediaPipe's
 * per-frame landmark noise briefly looks like a different gesture.
 */
function resolveStableMode(rawMode) {
    if (rawMode === stableMode) {
        pendingMode = null;
        pendingModeCount = 0;
        return stableMode;
    }

    if (rawMode === pendingMode) {
        pendingModeCount++;
    } else {
        pendingMode = rawMode;
        pendingModeCount = 1;
    }

    if (pendingModeCount >= HYSTERESIS_FRAMES) {
        stableMode = rawMode;
        pendingMode = null;
        pendingModeCount = 0;
    }

    return stableMode;
}

/* =========================================================================
 * MAIN LOOP
 * ========================================================================= */

/**
 * MediaPipe Hands results callback. Decides draw / erase / idle mode
 * (with hysteresis) and draws/erases using the smoothed fingertip point.
 */
function onResults(results) {
    if (isCameraPaused) return;
    updateFps();

    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
        resetStroke();
        const mode = resolveStableMode("IDLE");
        setMode(mode);
        if (landmarkCtx) landmarkCtx.clearRect(0, 0, landmarkCanvasEl.width, landmarkCanvasEl.height);
        updateDebugPanel({ mode, rawX: null, rawY: null, canvasX: null, canvasY: null, confidence: null });
        return;
    }

    const landmarks = results.multiHandLandmarks[0];
    const confidence =
        results.multiHandedness && results.multiHandedness[0]
            ? results.multiHandedness[0].score
            : null;

    drawLandmarksOverlay(landmarks);

    const fingersUp = getFingersUp(landmarks);
    const isIndexUp = fingersUp[1];
    const isMiddleUp = fingersUp[2];

    let rawMode = "IDLE";
    if (isIndexUp && isMiddleUp) rawMode = "ERASE";
    else if (isIndexUp && !isMiddleUp) rawMode = "DRAW";

    const mode = resolveStableMode(rawMode);
    setMode(mode);

    const indexTip = landmarks[8];
    const mapped = landmarkToCanvasCoords(indexTip);

    if (mode !== "IDLE" && mapped.visible) {
        const { x, y } = smoothPoint(mapped.x, mapped.y);
        if (mode === "ERASE") {
            eraseAt(x, y);
            resetStroke(); // don't let a later draw-stroke jump-connect from here
        } else if (mode === "DRAW") {
            drawLine(x, y);
        }
    } else {
        resetStroke();
    }

    updateDebugPanel({
        mode,
        rawX: indexTip.x,
        rawY: indexTip.y,
        canvasX: mapped.x,
        canvasY: mapped.y,
        confidence,
    });
}

function setupHands() {
    handsInstance = new Hands({
        locateFile: (file) =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    handsInstance.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.7,
    });

    handsInstance.onResults(onResults);
}

function setupCamera() {
    cameraInstance = new Camera(videoEl, {
        onFrame: async () => {
            if (isCameraPaused) return;
            await handsInstance.send({ image: videoEl });
        },
        width: 640,
        height: 480,
    });

    cameraInstance.start();
}

/**
 * Toggles the webcam between paused and running.
 * - Pausing: freezes the visible frame (videoEl.pause()) and stops
 *   sending frames to MediaPipe. The drawing canvas is left untouched.
 * - Resuming: resumes video playback and frame processing normally.
 */
export function toggleCamera() {
    isCameraPaused = !isCameraPaused;

    if (isCameraPaused) {
        videoEl.pause();
        resetStroke();
        setMode("IDLE");
    } else {
        videoEl.play();
        lastFrameTs = null; // avoid one huge FPS spike/dip right after resuming
    }
}

/**
 * Entry point, called on window load. Wires up video + canvas elements,
 * creates the debug overlay/panel, and starts hand tracking.
 */
export function initPainter() {
    videoEl = document.getElementById("video");
    canvasEl = document.getElementById("drawingCanvas");

    if (!videoEl || !canvasEl) {
        console.error("initPainter: #video or #drawingCanvas not found");
        return;
    }

    // Zoom/FaceTime-style mirror preview. This is the ONLY place the video
    // is ever CSS-transformed; the canvases are never transformed (see
    // warnIfCanvasMirrored) — mirroring for drawing/landmarks happens via
    // math in landmarkToCanvasCoords() instead, driven by MIRROR_CAMERA.
    videoEl.style.transform = MIRROR_CAMERA ? "scaleX(-1)" : "none";

    ctx = canvasEl.getContext("2d");

    createLandmarkCanvas();
    createDebugPanel();
    warnIfCanvasMirrored();

    videoEl.addEventListener("loadedmetadata", resizeCanvasToVideo);
    window.addEventListener("resize", resizeCanvasToVideo);

    // Handles the case where metadata was already available before this
    // script ran (the listener above would otherwise never fire).
    ensureCanvasSized();

    setupHands();
    setupCamera();
}

// Optional exports if you want manual control elsewhere (e.g. a Clear button)
export { clearCanvas, MIRROR_CAMERA };