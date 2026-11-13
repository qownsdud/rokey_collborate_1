(() => {
  const state = {
    camStream: null,
    camMirrored: true,
    capturedBlob: null,
    resultBlob: null,
    resultEngine: "",
    resultUrl: null,
    timerId: null,
  };

  const dlStatusPill = document.getElementById("dlStatusPill");

  const btnStartCam = document.getElementById("btnStartCam");
  const btnMirror = document.getElementById("btnMirror");
  const photoFileInput = document.getElementById("photoFileInput");
  const camVideo = document.getElementById("camVideo");
  const capturedPreview = document.getElementById("capturedPreview");
  const camPlaceholder = document.getElementById("camPlaceholder");
  const countdownOverlay = document.getElementById("countdownOverlay");
  const captureCanvas = document.getElementById("captureCanvas");
  const btnCaptureNow = document.getElementById("btnCaptureNow");
  const btnCaptureTimer = document.getElementById("btnCaptureTimer");
  const btnRetake = document.getElementById("btnRetake");

  const paperSelect = document.getElementById("paperSelect");
  const btnGenerate = document.getElementById("btnGenerate");
  const errorBanner = document.getElementById("errorBanner");

  const resultEmpty = document.getElementById("resultEmpty");
  const resultImage = document.getElementById("resultImage");
  const loadingOverlay = document.getElementById("loadingOverlay");
  const loadingText = document.getElementById("loadingText");
  const loadingTimer = document.getElementById("loadingTimer");
  const engineBadge = document.getElementById("engineBadge");
  const metaBar = document.getElementById("metaBar");
  const metaTime = document.getElementById("metaTime");
  const metaSize = document.getElementById("metaSize");
  const metaBlack = document.getElementById("metaBlack");
  const btnConfirmSelect = document.getElementById("btnConfirmSelect");
  const btnResetAll = document.getElementById("btnResetAll");
  const confirmBanner = document.getElementById("confirmBanner");

  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      if (data.dl_ready) {
        dlStatusPill.textContent = "로컬 DL: 준비됨";
        dlStatusPill.className = "status-pill ok";
      } else {
        dlStatusPill.textContent = "로컬 DL: 미설치";
        dlStatusPill.className = "status-pill warn";
      }
    } catch {
      dlStatusPill.textContent = "서버 연결 오류";
      dlStatusPill.className = "status-pill warn";
    }
  }

  function showError(msg) {
    errorBanner.textContent = msg;
    errorBanner.hidden = false;
  }

  function clearError() {
    errorBanner.hidden = true;
    errorBanner.textContent = "";
  }

  function applyMirrorClass() {
    camVideo.classList.toggle("mirrored", state.camMirrored);
    btnMirror.classList.toggle("active", state.camMirrored);
    btnMirror.textContent = state.camMirrored ? "좌우 반전 ON" : "좌우 반전 OFF";
  }

  async function startCamera() {
    clearError();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError("이 브라우저에서는 카메라 API를 지원하지 않습니다. 사진 파일을 선택해 주세요.");
      return;
    }
    try {
      stopCamera();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      state.camStream = stream;
      camVideo.srcObject = stream;
      camVideo.hidden = false;
      capturedPreview.hidden = true;
      camPlaceholder.hidden = true;
      state.capturedBlob = null;
      btnCaptureNow.disabled = false;
      btnCaptureTimer.disabled = false;
      btnRetake.hidden = true;
      btnStartCam.textContent = "카메라 켜짐";
      applyMirrorClass();
    } catch (err) {
      showError(`카메라를 열 수 없습니다: ${err.message || "권한을 확인해 주세요."}`);
    }
  }

  function stopCamera() {
    if (state.camStream) {
      state.camStream.getTracks().forEach((t) => t.stop());
      state.camStream = null;
    }
    btnStartCam.textContent = "카메라 켜기";
  }

  function takeSnapshot() {
    if (!state.camStream || camVideo.videoWidth === 0) return;
    const vw = camVideo.videoWidth;
    const vh = camVideo.videoHeight;
    const side = Math.min(vw, vh);
    const sx = (vw - side) / 2;
    const sy = (vh - side) / 2;

    captureCanvas.width = 768;
    captureCanvas.height = 768;
    const cctx = captureCanvas.getContext("2d");
    cctx.save();
    if (state.camMirrored) {
      cctx.translate(768, 0);
      cctx.scale(-1, 1);
    }
    cctx.drawImage(camVideo, sx, sy, side, side, 0, 0, 768, 768);
    cctx.restore();

    captureCanvas.toBlob(
      (blob) => {
        if (!blob) return;
        state.capturedBlob = blob;
        capturedPreview.src = URL.createObjectURL(blob);
        capturedPreview.hidden = false;
        camVideo.hidden = true;
        camPlaceholder.hidden = true;
        btnRetake.hidden = false;
        btnCaptureNow.disabled = true;
        btnCaptureTimer.disabled = true;
      },
      "image/jpeg",
      0.92
    );
  }

  btnStartCam.addEventListener("click", startCamera);
  btnMirror.addEventListener("click", () => {
    state.camMirrored = !state.camMirrored;
    applyMirrorClass();
  });

  btnCaptureNow.addEventListener("click", takeSnapshot);

  btnCaptureTimer.addEventListener("click", () => {
    if (!state.camStream) return;
    let count = 3;
    countdownOverlay.textContent = String(count);
    countdownOverlay.hidden = false;
    btnCaptureNow.disabled = true;
    btnCaptureTimer.disabled = true;

    const interval = setInterval(() => {
      count -= 1;
      if (count <= 0) {
        clearInterval(interval);
        countdownOverlay.hidden = true;
        takeSnapshot();
      } else {
        countdownOverlay.textContent = String(count);
      }
    }, 1000);
  });

  btnRetake.addEventListener("click", () => {
    state.capturedBlob = null;
    capturedPreview.hidden = true;
    if (state.camStream) {
      camVideo.hidden = false;
      btnCaptureNow.disabled = false;
      btnCaptureTimer.disabled = false;
    } else {
      camPlaceholder.hidden = false;
    }
    btnRetake.hidden = true;
  });

  photoFileInput.addEventListener("change", () => {
    const file = photoFileInput.files?.[0];
    if (!file) return;
    clearError();
    state.capturedBlob = file;
    capturedPreview.src = URL.createObjectURL(file);
    capturedPreview.hidden = false;
    camVideo.hidden = true;
    camPlaceholder.hidden = true;
    btnRetake.hidden = false;
  });

  function startLoading(message) {
    const startedAt = performance.now();
    loadingText.textContent = message;
    loadingTimer.textContent = "0.0초 경과";
    loadingOverlay.hidden = false;
    btnGenerate.disabled = true;
    state.timerId = setInterval(() => {
      const sec = ((performance.now() - startedAt) / 1000).toFixed(1);
      loadingTimer.textContent = `${sec}초 경과`;
    }, 100);
  }

  function stopLoading() {
    loadingOverlay.hidden = true;
    btnGenerate.disabled = false;
    if (state.timerId) {
      clearInterval(state.timerId);
      state.timerId = null;
    }
  }

  btnGenerate.addEventListener("click", async () => {
    clearError();
    if (!state.capturedBlob) {
      showError("먼저 노트북 카메라로 사진을 촬영하거나 사진 파일을 선택해 주세요.");
      return;
    }

    const detail =
      document.querySelector('input[name="dlDetail"]:checked')?.value || "medium";
    const formData = new FormData();
    formData.append("photo", state.capturedBlob, "webcam_photo.jpg");
    formData.append("detail", detail);
    formData.append("paper", paperSelect.value);

    startLoading("로컬 딥러닝 모델로 선화를 추출하고 있습니다...");

    try {
      const res = await fetch("/api/kiosk/photo", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        let msg = `요청 실패 (HTTP ${res.status})`;
        try {
          const errJson = await res.json();
          if (errJson.detail) msg = errJson.detail;
        } catch {
          // ignore
        }
        throw new Error(msg);
      }

      const blob = await res.blob();
      if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
      state.resultBlob = blob;
      state.resultUrl = URL.createObjectURL(blob);

      resultImage.src = state.resultUrl;
      resultImage.hidden = false;
      resultEmpty.hidden = true;
      confirmBanner.hidden = true;

      const elapsed = res.headers.get("X-Elapsed-Sec") || "-";
      const imgSize = res.headers.get("X-Image-Size") || "-";
      const blackRatio = res.headers.get("X-Black-Ratio");
      const engine = res.headers.get("X-Engine") || "Local DL";
      state.resultEngine = engine;

      engineBadge.textContent = engine;
      engineBadge.hidden = false;
      metaTime.textContent = elapsed === "-" ? "-" : `${elapsed}초`;
      metaSize.textContent = imgSize;
      metaBlack.textContent = blackRatio
        ? `${(Number(blackRatio) * 100).toFixed(1)}%`
        : "-";
      metaBar.hidden = false;

      btnConfirmSelect.disabled = false;
      btnConfirmSelect.textContent = "✅ 사진 최종 선택";
    } catch (err) {
      showError(err.message || "라인아트 생성 중 오류가 발생했습니다.");
    } finally {
      stopLoading();
    }
  });

  btnConfirmSelect.addEventListener("click", async () => {
    if (!state.resultBlob) return;
    clearError();
    btnConfirmSelect.disabled = true;
    btnConfirmSelect.textContent = "저장 중...";

    try {
      const formData = new FormData();
      formData.append("image", state.resultBlob, "selected_lineart.png");
      formData.append("paper", paperSelect.value);
      formData.append("engine", state.resultEngine || "Local DL");

      const res = await fetch("/api/kiosk/confirm", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "사진 최종 선택 저장에 실패했습니다.");
      }

      confirmBanner.innerHTML =
        `✅ <strong>${data.message}</strong><br />` +
        `로봇 드로잉 경로: <code>${data.latest_file}</code>`;
      confirmBanner.hidden = false;
      btnConfirmSelect.textContent = "✅ 최종 선택 완료";
    } catch (err) {
      showError(err.message || "최종 선택 처리 중 오류가 발생했습니다.");
      btnConfirmSelect.disabled = false;
      btnConfirmSelect.textContent = "✅ 사진 최종 선택";
    }
  });

  btnResetAll.addEventListener("click", () => {
    clearError();
    state.resultBlob = null;
    state.resultEngine = "";
    resultImage.hidden = true;
    resultEmpty.hidden = false;
    metaBar.hidden = true;
    engineBadge.hidden = true;
    confirmBanner.hidden = true;
    btnConfirmSelect.disabled = true;
    btnConfirmSelect.textContent = "✅ 사진 최종 선택";
    btnRetake.click();
  });

  applyMirrorClass();
  checkHealth();
})();
