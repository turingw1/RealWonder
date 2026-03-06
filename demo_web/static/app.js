// RealWonder Interactive Demo - Frontend JavaScript

(function() {
    "use strict";

    const socket = io();
    const canvas = document.getElementById("videoCanvas");
    const ctx = canvas.getContext("2d");

    const loadingPanel = document.getElementById("loadingPanel");
    const controlsPanel = document.getElementById("controlsPanel");
    const caseNameEl = document.getElementById("caseName");
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
    const resetBtn = document.getElementById("resetBtn");
    const statusBar = document.getElementById("statusBar");
    const progressBar = document.getElementById("progressBar");
    const progressText = document.getElementById("progressText");
    const frameCounter = document.getElementById("frameCounter");
    const forceConfigContainer = document.getElementById("forceConfigContainer");
    const objectSelect = document.getElementById("objectSelect");
    const forceViewerWrap = document.getElementById("forceViewerWrap");
    const videoPanel = document.getElementById("videoPanel");

    let frameCount = 0;
    let isGenerating = false;
    let allowChangeForce = false;

    // Per-object state: { viewX, viewY, viewZ, maxStrength }
    // viewX = screen right(+) / left(-)
    // viewY = screen up(+)    / down(-)
    // viewZ = out of screen(+)/ into screen(-)
    let objectForces = [];
    let objectConfigs = [];
    let objectCentroids = [];  // [{x, y}] in canvas pixel coords
    let maskImages = [];
    let previewImage = null;
    let selectedObjectIdx = -1;

    // ---- Three.js 3D viewer ----
    var renderer3d = null;
    var scene3d = null;
    var camera3d = null;
    var forceArrowGroup3d = null;
    var viewer3dReady = false;

    // Axis config — display coordinate system: X=right, Y=up, Z=out of screen
    const AXES = [
        { key: "viewX", name: "X — Left / Right",         neg: "← Left",          pos: "Right →",         color: "#cc2222" },
        { key: "viewY", name: "Y — Down / Up",             neg: "↓ Down",          pos: "Up ↑",            color: "#22aa44" },
        { key: "viewZ", name: "Z — Into / Out of screen",  neg: "◀ Into screen",   pos: "Out of screen ▶", color: "#00ccdd" },
    ];

    // ---- SocketIO ----

    socket.on("connect", function() { setStatus("Connected \u2014 waiting for server data..."); });
    socket.on("disconnect", function() { setStatus("Disconnected from server"); });

    socket.on("ready", function(data) {
        setStatus("Ready. Set force direction with the sliders, then click Start.");
        caseNameEl.textContent = "Case: " + (data.case_name || "");

        if (data.preview) {
            previewImage = new Image();
            previewImage.onload = function() { redrawCanvas(); };
            previewImage.src = "data:image/jpeg;base64," + data.preview;
        }
        if (data.ui_config) {
            allowChangeForce = !!data.ui_config.allow_change_force;
            buildForceControls(data.ui_config);
        }

        loadingPanel.style.display = "none";
        controlsPanel.style.display = "block";
        enableControls(true);
    });

    socket.on("frame", function(data) {
        frameCount++;
        frameCounter.textContent = "Frame: " + frameCount;
        var img = new Image();
        img.onload = function() { ctx.drawImage(img, 0, 0, canvas.width, canvas.height); };
        img.src = "data:image/jpeg;base64," + data.data;
    });

    socket.on("status", function(data) {
        setStatus(data.message);
        if (data.block !== undefined && data.total_blocks !== undefined) {
            var pct = ((data.block + 1) / data.total_blocks * 100).toFixed(0);
            progressBar.style.width = pct + "%";
            progressText.textContent = "Block " + (data.block + 1) + "/" + data.total_blocks;
        }
    });

    socket.on("error", function(data) { setStatus("Error: " + data.message); });

    socket.on("generation_complete", function() {
        setStatus("Generation complete. Click Reset to run again.");
        progressBar.style.width = "100%";
        isGenerating = false;
        enableControls(true);
        showForceViewer(true);
    });

    // ---- Centroid from mask ----

    function computeMaskCentroid(maskImg) {
        var off = document.createElement("canvas");
        off.width = canvas.width; off.height = canvas.height;
        var offCtx = off.getContext("2d");
        offCtx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
        var data = offCtx.getImageData(0, 0, canvas.width, canvas.height).data;
        var sx = 0, sy = 0, n = 0;
        for (var i = 0; i < data.length; i += 4) {
            if (data[i] > 128) {
                sx += (i / 4) % canvas.width;
                sy += Math.floor((i / 4) / canvas.width);
                n++;
            }
        }
        return n === 0
            ? { x: canvas.width / 2, y: canvas.height / 2 }
            : { x: sx / n, y: sy / n };
    }

    // ---- Build controls (once per ready) ----

    function buildForceControls(uiConfig) {
        objectConfigs = uiConfig.objects || [];
        objectForces = [];
        objectCentroids = [];
        maskImages = [];
        objectSelect.innerHTML = "";

        for (var i = 0; i < objectConfigs.length; i++) {
            var obj = objectConfigs[i];
            objectForces.push({ viewX: 0, viewY: 0, viewZ: 0, maxStrength: obj.max_strength || 2.0 });
            objectCentroids.push({ x: canvas.width / 2, y: canvas.height / 2 });

            var maskImg = null;
            if (obj.mask_b64) {
                maskImg = new Image();
                (function(idx, img) {
                    img.onload = function() {
                        objectCentroids[idx] = computeMaskCentroid(img);
                        if (idx === selectedObjectIdx) {
                            positionForceViewer(idx);
                        }
                        redrawCanvas();  // always redraw — harmless if not selected
                    };
                    img.src = "data:image/png;base64," + obj.mask_b64;
                })(i, maskImg);
            }
            maskImages.push(maskImg);

            var opt = document.createElement("option");
            opt.value = i; opt.textContent = obj.label || ("Object " + obj.idx);
            objectSelect.appendChild(opt);
        }

        objectSelect.onchange = function() {
            showObjectControls(parseInt(objectSelect.value));
        };

        setup3DViewer();

        if (objectConfigs.length > 0) {
            selectedObjectIdx = 0;
            objectSelect.value = "0";
            showObjectControls(0);
        }
    }

    // ---- Per-object controls ----

    function showObjectControls(idx) {
        selectedObjectIdx = idx;
        redrawCanvas();
        positionForceViewer(idx);
        forceConfigContainer.innerHTML = "";
        if (idx < 0 || idx >= objectConfigs.length) return;

        var obj = objectConfigs[idx];
        var force = objectForces[idx];
        var maxStr = force.maxStrength;

        var group = document.createElement("div");
        group.className = "object-force-group";

        var titleEl = document.createElement("div");
        titleEl.className = "object-force-label";
        titleEl.textContent = obj.label || ("Object " + obj.idx);
        group.appendChild(titleEl);

        // Force summary
        var summary = document.createElement("div");
        summary.id = "forceSummary_" + idx;
        summary.className = "force-summary force-none";
        summary.textContent = "No force set";
        group.appendChild(summary);

        // Three axis sliders
        AXES.forEach(function(ax) {
            group.appendChild(makeAxisSlider(idx, ax, maxStr, force[ax.key]));
        });

        // Reset force button
        var resetForceBtn = document.createElement("button");
        resetForceBtn.className = "reset-force-btn";
        resetForceBtn.textContent = "Reset force to zero";
        resetForceBtn.addEventListener("click", function() {
            objectForces[idx].viewX = 0;
            objectForces[idx].viewY = 0;
            objectForces[idx].viewZ = 0;
            AXES.forEach(function(ax) {
                var sl = document.getElementById("axisSlider_" + idx + "_" + ax.key);
                var vl = document.getElementById("axisVal_" + idx + "_" + ax.key);
                if (sl) sl.value = "0";
                if (vl) vl.textContent = "0.0";
            });
            update3DArrow(0, 0, 0);
        updateForceSummary(idx);
            if (isGenerating && allowChangeForce) emitForceUpdate();
        });
        group.appendChild(resetForceBtn);

        forceConfigContainer.appendChild(group);
        updateForceSummary(idx);
        update3DArrow(force.viewX, force.viewY, force.viewZ);
    }

    function makeAxisSlider(objIdx, ax, maxStr, initialValue) {
        var group = document.createElement("div");
        group.className = "axis-slider-group";

        // Header row: dot + name + value
        var header = document.createElement("div");
        header.className = "axis-slider-header";

        var dot = document.createElement("span");
        dot.className = "axis-dot";
        dot.style.background = ax.color;
        header.appendChild(dot);

        var name = document.createElement("span");
        name.className = "axis-name";
        name.textContent = ax.name;
        header.appendChild(name);

        var valEl = document.createElement("span");
        valEl.className = "axis-value";
        valEl.id = "axisVal_" + objIdx + "_" + ax.key;
        valEl.textContent = (initialValue || 0).toFixed(1);
        header.appendChild(valEl);
        group.appendChild(header);

        // Endpoint labels
        var endLabels = document.createElement("div");
        endLabels.className = "axis-end-labels";
        endLabels.innerHTML = "<span>" + ax.neg + "</span><span>" + ax.pos + "</span>";
        group.appendChild(endLabels);

        // Slider
        var slider = document.createElement("input");
        slider.type = "range";
        slider.min = String(-maxStr);
        slider.max = String(maxStr);
        slider.step = "0.1";
        slider.value = String(initialValue || 0);
        slider.id = "axisSlider_" + objIdx + "_" + ax.key;
        slider.addEventListener("input", function() {
            var val = parseFloat(slider.value);
            objectForces[objIdx][ax.key] = val;
            valEl.textContent = val.toFixed(1);
            var f = objectForces[objIdx];
            update3DArrow(f.viewX, f.viewY, f.viewZ);
            updateForceSummary(objIdx);
            if (isGenerating && allowChangeForce) emitForceUpdate();
        });
        group.appendChild(slider);

        return group;
    }

    // ---- Force summary ----

    function updateForceSummary(idx) {
        var el = document.getElementById("forceSummary_" + idx);
        if (!el) return;
        var f = objectForces[idx];
        var str = getTotalStrength(f);
        var dir = get3DDirection(f);
        if (str < 0.01) {
            el.textContent = "No force set";
            el.className = "force-summary force-none";
        } else {
            el.innerHTML =
                "<span class='force-strength'>" + str.toFixed(1) + " units</span>" +
                "<span class='force-vec'>[" +
                dir[0].toFixed(2) + ", " + dir[1].toFixed(2) + ", " + dir[2].toFixed(2) + "]</span>";
            el.className = "force-summary force-active";
        }
    }

    // ---- 3D force helpers ----

    function get3DDirection(force) {
        var len = Math.sqrt(force.viewX*force.viewX + force.viewY*force.viewY + force.viewZ*force.viewZ);
        if (len < 1e-6) return [0, 0, 0];
        // Convert view coords (X=right, Y=up, Z=out) → server world coords (forceX=right, forceY=in, forceZ=up)
        // forceX = viewX,  forceY = -viewZ (server in = viewer into-screen = negative out),  forceZ = viewY
        return [force.viewX / len, -force.viewZ / len, force.viewY / len];
    }

    function getTotalStrength(force) {
        var raw = Math.sqrt(force.viewX*force.viewX + force.viewY*force.viewY + force.viewZ*force.viewZ);
        return Math.min(raw, force.maxStrength || 2.0);
    }

    // ---- Three.js 3D viewer ----

    // CylinderGeometry shaft + ConeGeometry head — required because WebGL ignores linewidth > 1.
    function makeFatArrow(dir, totalLength, shaftRadius, headLength, headRadius, color) {
        var mat      = new THREE.MeshBasicMaterial({ color: color });
        var shaftLen = Math.max(totalLength - headLength, 0.001);
        var group    = new THREE.Group();

        var shaft = new THREE.Mesh(new THREE.CylinderGeometry(shaftRadius, shaftRadius, shaftLen, 12), mat);
        shaft.position.y = shaftLen / 2;
        group.add(shaft);

        var head = new THREE.Mesh(new THREE.ConeGeometry(headRadius, headLength, 12), mat.clone());
        head.position.y = shaftLen + headLength / 2;
        group.add(head);

        var norm = dir.clone().normalize();
        var up   = new THREE.Vector3(0, 1, 0);
        if (Math.abs(norm.dot(up)) > 0.9999) {
            if (norm.y < 0) group.rotation.z = Math.PI;
        } else {
            group.setRotationFromQuaternion(new THREE.Quaternion().setFromUnitVectors(up, norm));
        }
        return group;
    }

    function setup3DViewer() {
        if (typeof THREE === "undefined") {
            console.warn("Three.js not loaded — 3D viewer disabled.");
            return;
        }
        var viewerCanvas = document.getElementById("forceViewer3d");
        if (!viewerCanvas) return;

        renderer3d = new THREE.WebGLRenderer({ canvas: viewerCanvas, antialias: true, alpha: true });
        renderer3d.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer3d.setSize(160, 160);
        renderer3d.setClearColor(0x000000, 0);  // fully transparent GL background — CSS controls the tint

        scene3d = new THREE.Scene();

        camera3d = new THREE.PerspectiveCamera(42, 1.0, 0.1, 50);
        // Camera along (1,1,1): symmetric view — X goes right, Y goes up, Z goes lower-left (out of screen)
        camera3d.position.set(0.37, 0.69, 2.66);
        camera3d.lookAt(0, 0, 0);

        // Origin sphere
        var originMesh = new THREE.Mesh(
            new THREE.SphereGeometry(0.06, 16, 16),
            new THREE.MeshBasicMaterial({ color: 0xffffff })
        );
        scene3d.add(originMesh);

        // Reference axis arrows — X=red, Y=green, Z=cyan
        // totalLength=0.9, shaftRadius=0.04, headLength=0.18, headRadius=0.06
        var axLen = 0.9, axSR = 0.024, axHL = 0.216, axHR = 0.072;
        scene3d.add(makeFatArrow(new THREE.Vector3(1,0,0), axLen, axSR, axHL, axHR, 0xcc2222));
        scene3d.add(makeFatArrow(new THREE.Vector3(0,1,0), axLen, axSR, axHL, axHR, 0x22aa44));
        scene3d.add(makeFatArrow(new THREE.Vector3(0,0,1), axLen, axSR, axHL, axHR, 0x00ccdd));

        scene3d.add(new THREE.AmbientLight(0xffffff, 1.0));

        viewer3dReady = true;
        render3D();
    }

    // viewX=right, viewY=up, viewZ=out of screen — map directly to Three.js axes
    function update3DArrow(viewX, viewY, viewZ) {
        if (!viewer3dReady) return;

        if (forceArrowGroup3d) { scene3d.remove(forceArrowGroup3d); forceArrowGroup3d = null; }

        var len = Math.sqrt(viewX*viewX + viewY*viewY + viewZ*viewZ);
        if (len > 0.01) {
            var maxStr = (selectedObjectIdx >= 0 && objectForces[selectedObjectIdx])
                ? objectForces[selectedObjectIdx].maxStrength : 2.0;
            var arrowLen = 0.9 * Math.min(len / maxStr, 1.0);
            forceArrowGroup3d = makeFatArrow(
                new THREE.Vector3(viewX/len, viewY/len, viewZ/len),
                arrowLen, 0.024, 0.216, 0.072, 0xff6600
            );
            scene3d.add(forceArrowGroup3d);
        }
        render3D();
    }

    function render3D() {
        if (renderer3d && scene3d && camera3d) {
            renderer3d.render(scene3d, camera3d);
        }
    }

    // Position the viewer wrap at the top-left corner of the video panel
    function positionForceViewer(idx) {
        if (!forceViewerWrap) return;

        forceViewerWrap.style.left = "50px";
        forceViewerWrap.style.top  = "50px";
        forceViewerWrap.style.display = "flex";
    }

    function showForceViewer(visible) {
        if (!forceViewerWrap) return;
        if (visible) {
            positionForceViewer(selectedObjectIdx);
        } else {
            forceViewerWrap.style.display = "none";
        }
    }

    // ---- Canvas drawing (preview + mask overlay only — no force arrow on canvas) ----

    function redrawCanvas() {
        if (isGenerating) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (previewImage && previewImage.complete) {
            ctx.drawImage(previewImage, 0, 0, canvas.width, canvas.height);
        } else {
            ctx.fillStyle = "#e8ecf2";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }

        var idx = selectedObjectIdx;
        if (idx >= 0 && maskImages[idx] && maskImages[idx].complete) {
            drawMaskOverlay(maskImages[idx]);
        }
    }

    function drawMaskOverlay(maskImg) {
        var off = document.createElement("canvas");
        off.width = canvas.width; off.height = canvas.height;
        var offCtx = off.getContext("2d");
        offCtx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
        var d = offCtx.getImageData(0, 0, canvas.width, canvas.height);
        for (var i = 0; i < d.data.length; i += 4) {
            if (d.data[i] > 128) {
                d.data[i] = 80; d.data[i+1] = 140; d.data[i+2] = 237; d.data[i+3] = 100;
            } else {
                d.data[i+3] = 0;
            }
        }
        offCtx.putImageData(d, 0, 0);
        ctx.drawImage(off, 0, 0);
    }

    // ---- Collect and emit ----

    function collectForces() {
        return objectForces.map(function(f, i) {
            return {
                obj_idx: i,
                direction: get3DDirection(f),
                strength: getTotalStrength(f),
            };
        });
    }

    function emitForceUpdate() {
        socket.emit("update_forces", { forces: collectForces() });
    }

    // ---- Start / Stop / Reset ----

    startBtn.addEventListener("click", function() {
        frameCount = 0;
        frameCounter.textContent = "Frame: 0";
        progressBar.style.width = "0%";
        progressText.textContent = "";
        isGenerating = true;
        enableControls(false);
        socket.emit("start_generation", {
            forces: collectForces(),
        });
    });

    stopBtn.addEventListener("click", function() { socket.emit("stop_generation"); });

    resetBtn.addEventListener("click", function() {
        frameCount = 0;
        frameCounter.textContent = "Frame: 0";
        progressBar.style.width = "0%";
        progressText.textContent = "";
        isGenerating = false;
        socket.emit("reset");
    });

    document.addEventListener("keydown", function(e) {
        if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
        if (e.target.tagName === "SELECT") return;
        if (e.key === "Enter") startBtn.click();
    });

    // ---- Helpers ----

    function enableControls(enabled) {
        startBtn.disabled = !enabled;
        var forceEnabled = enabled || (isGenerating && allowChangeForce);
        objectSelect.disabled = !forceEnabled;
        var sliders = forceConfigContainer.querySelectorAll("input[type='range']");
        for (var j = 0; j < sliders.length; j++) sliders[j].disabled = !forceEnabled;
        var resetBtns = forceConfigContainer.querySelectorAll(".reset-force-btn");
        for (var k = 0; k < resetBtns.length; k++) resetBtns[k].disabled = !forceEnabled;
    }

    function setStatus(msg) { statusBar.textContent = "Status: " + msg; }

    // Placeholder
    ctx.fillStyle = "#e8ecf2";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#9ca3b0";
    ctx.font = "500 16px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Initializing...", canvas.width / 2, canvas.height / 2);

})();
