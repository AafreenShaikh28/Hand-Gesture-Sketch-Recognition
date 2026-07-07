// prediction.js
// Handles periodic capture of the drawing canvas and sending it
// to the FastAPI /predict endpoint, then rendering the guesses.
import { PREDICT_URL } from "./backend/congif.js";
// ---- State ----
let guesses = [];       // stores latest prediction results
let isPredicting = false; // prevents overlapping requests
let predictionIntervalId = null;

const INTERVAL_MS = 5000;

/**
 * Checks whether the given canvas has anything drawn on it.
 * Works by reading pixel data and checking if any pixel has
 * non-zero alpha (assuming a transparent canvas background).
 * If your canvas has an opaque background instead, adjust the
 * check to compare against that background color.
 */
function isCanvasBlank(canvas) {
    const ctx = canvas.getContext("2d");
    const { width, height } = canvas;

    if (width === 0 || height === 0) return true;

    const pixelData = ctx.getImageData(0, 0, width, height).data;

    // Check alpha channel (every 4th byte) for any non-zero value
    for (let i = 3; i < pixelData.length; i += 4) {
        if (pixelData[i] !== 0) {
            return false; // found a drawn (non-transparent) pixel
        }
    }
    return true; // no drawn pixels found
}

/**
 * Captures the current canvas content as a PNG Blob.
 * Returns a Promise<Blob>.
 */
function captureCanvas(canvas) {
    return new Promise((resolve, reject) => {
        canvas.toBlob((blob) => {
            if (blob) {
                resolve(blob);
            } else {
                reject(new Error("Failed to convert canvas to Blob"));
            }
        }, "image/png");
    });
}

/**
 * Sends the captured Blob to the FastAPI backend and returns
 * the parsed JSON response (or null on failure).
 */
async function sendPrediction(blob) {
    const formData = new FormData();
    formData.append("file", blob, "drawing.png");

    try {
        const response = await fetch(PREDICT_URL, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Server responded with status ${response.status}`);
        }

        const data = await response.json();
        return data;
    } catch (err) {
        console.error("Prediction request failed:", err);
        return null;
    }
}

/**
 * Updates the existing ".pred" container with the latest guesses.
 * Does NOT alter your HTML structure/CSS file — it only injects
 * a results block dynamically inside the existing container.
 *
 * Markup built per guess:
 *   .guess-item (+ .guess-item--top on the highest-confidence guess)
 *     .guess-row
 *       .guess-label
 *       .guess-confidence
 *     .guess-bar-track
 *       .guess-bar-fill  (width animated in on the next frame)
 *
 * See prediction-panel.css for all visual styling.
 */
function renderGuesses(guessList) {
    const container = document.querySelector(".pred");
    if (!container) {
        console.warn('renderGuesses: ".pred" container not found');
        return;
    }

    // Create (or reuse) a dedicated results element so we don't
    // wipe out the existing <img class="guess-img"> in .pred
    let resultsEl = container.querySelector("#guessResults");
    if (!resultsEl) {
        resultsEl = document.createElement("div");
        resultsEl.id = "guessResults";
        container.appendChild(resultsEl);
    }

    if (!guessList || guessList.length === 0) {
        resultsEl.innerHTML = "";
        return;
    }

    resultsEl.innerHTML = guessList
        .map((g, index) => {
            const pct = (g.confidence * 100).toFixed(1);
            const isTop = index === 0;
            return `<div class="guess-item${isTop ? " guess-item--top" : ""}">
                    <div class="guess-row">
                        <span class="guess-label">${g.label}</span>
                        <span class="guess-confidence">${pct}%</span>
                    </div>
                    <div class="guess-bar-track">
                        <div class="guess-bar-fill" data-target-width="${pct}"></div>
                    </div>
                 </div>`;
        })
        .join("");

    // Animate bars from 0 -> target width. Setting the width in the same
    // frame the elements are created would skip the CSS transition, so we
    // wait one animation frame before applying the real width.
    requestAnimationFrame(() => {
        resultsEl.querySelectorAll(".guess-bar-fill").forEach((el) => {
            el.style.width = `${el.dataset.targetWidth}%`;
        });
    });
}

/**
 * Main loop tick: captures canvas, checks blank state,
 * sends prediction, updates guesses + UI.
 */
async function predictionTick(canvas) {
    if (isPredicting) return; // skip if a request is still pending

    if (isCanvasBlank(canvas)) {
        // Nothing drawn — clear guesses and skip sending a request
        guesses = [];
        renderGuesses(guesses);
        return;
    }

    isPredicting = true;
    try {
        const blob = await captureCanvas(canvas);
        const result = await sendPrediction(blob);

        if (result && Array.isArray(result.guesses)) {
            guesses = result.guesses;
            renderGuesses(guesses);
        }
    } catch (err) {
        console.error("predictionTick error:", err);
    } finally {
        isPredicting = false;
    }
}

/**
 * Starts the 5-second prediction loop using the given canvas element.
 */
function startPredictionLoop(canvas) {
    if (predictionIntervalId !== null) {
        clearInterval(predictionIntervalId); // avoid duplicate loops
    }
    predictionIntervalId = setInterval(() => {
        predictionTick(canvas);
    }, INTERVAL_MS);
}

/**
 * Entry point — call this once the page/canvas is ready.
 */
function initPrediction() {
    const canvas = document.getElementById("drawingCanvas");
    if (!canvas) {
        console.error('initPrediction: canvas with id "drawingCanvas" not found');
        return;
    }
    startPredictionLoop(canvas);
}

// Auto-start once the page loads
window.addEventListener("load", initPrediction);

// Export in case you want to trigger/stop manually elsewhere
export { initPrediction, startPredictionLoop, renderGuesses, isCanvasBlank };