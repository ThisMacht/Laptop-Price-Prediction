const fields = [
  "brand",
  "model",
  "ram_gb",
  "storage_gb",
  "storage_type",
  "screen_size_inch",
  "cpu_text",
  "cpu_brand",
  "cpu_family",
  "cpu_generation",
  "cpu_suffix",
  "gpu_text",
  "condition",
  "warranty_status",
];

const example = {
  description:
    "Dell Inspiron 15, Intel Core i5-1235U, RAM 16GB, SSD 512GB, màn 15.6 inch, card Intel Integrated, đã sử dụng chưa sửa chữa, hết bảo hành",
  compareDescription: `Laptop A:
RAM 8GB, SSD 512GB, CPU Intel Core i5, đã sử dụng, giá 12.000.000 VND

Laptop B:
RAM 16GB, SSD 512GB, CPU Intel Core i5, đã sử dụng, giá 14.000.000 VND

Laptop C:
RAM 16GB, SSD 1TB, CPU Intel Core i7, đã sử dụng, giá 18.000.000 VND`,
  raw_features: {
    brand: "Dell",
    model: "Inspiron",
    ram_gb: 16,
    storage_gb: 512,
    storage_type: "SSD",
    screen_size_inch: 15.6,
    cpu_text: "Intel Core i5-1235U",
    cpu_brand: "Intel",
    cpu_family: "Intel Core i5",
    cpu_generation: 12,
    cpu_suffix: "U",
    gpu_text: "Intel Integrated",
    condition: "good",
    warranty_status: "expired",
  },
};

const form = document.querySelector("#predictForm");
const runState = document.querySelector("#runState");
const submitButton = document.querySelector("#submitButton");
const descriptionLabel = document.querySelector("#descriptionLabel");
const descriptionInput = document.querySelector("#description");
const predictedPriceRange = document.querySelector("#predictedPriceRange");
const predictedPricePoint = document.querySelector("#predictedPricePoint");
const completenessBadge = document.querySelector("#completenessBadge");
const uncertaintyBadge = document.querySelector("#uncertaintyBadge");
const compareBestPick = document.querySelector("#compareBestPick");
const compareSummary = document.querySelector("#compareSummary");
const compareTableBody = document.querySelector("#compareTableBody");
const compareValidationList = document.querySelector("#compareValidationList");
const compareResultsSection = document.querySelector("#compareResults");
const compareRunState = document.querySelector("#compareRunState");
const configChartSection = document.querySelector("#configChartSection");
const configChartPanel = document.querySelector("#configChartPanel");
const configChartState = document.querySelector("#configChartState");
const healthStatus = document.querySelector("#healthStatus");
const inputPanel = document.querySelector(".input-panel");
const resultPanel = document.querySelector(".result-panel");
const studioGrid = document.querySelector("#input");
let activeApiBase = null;

function setHealth(status) {
  if (healthStatus) healthStatus.textContent = status;
}

function apiBaseCandidates() {
  const candidates = [];
  const isHttpPage = window.location.protocol === "http:" || window.location.protocol === "https:";

  if (isHttpPage) candidates.push("");
  candidates.push("http://127.0.0.1:8000");
  candidates.push("http://localhost:8000");

  if (
    isHttpPage &&
    window.location.hostname &&
    !["127.0.0.1", "localhost"].includes(window.location.hostname)
  ) {
    candidates.push(`http://${window.location.hostname}:8000`);
  }

  return [...new Set(candidates)];
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 5000) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    window.clearTimeout(timeout);
  }
}

async function apiRequest(path, options = {}, timeoutMs = 60000) {
  const bases = activeApiBase === null ? apiBaseCandidates() : [activeApiBase, ...apiBaseCandidates()];
  const errors = [];

  for (const base of [...new Set(bases)]) {
    try {
      const response = await fetchWithTimeout(`${base}${path}`, options, timeoutMs);
      if (!response.ok) {
        const body = await response.text();
        throw new Error(`${response.status} ${response.statusText}${body ? `: ${body}` : ""}`);
      }
      activeApiBase = base;
      return response;
    } catch (error) {
      errors.push(`${base || "same-origin"} -> ${error.message}`);
    }
  }

  throw new Error(errors.join(" | "));
}

function getMode() {
  return document.querySelector('input[name="mode"]:checked')?.value || "text";
}

function getTask() {
  const checked = document.querySelector('input[name="task"]:checked');
  if (checked) return checked.value;
  return form?.dataset.activeTask || "predict";
}

function isCompareTask() {
  return getMode() === "text" && getTask() === "compare";
}

function isCompareResponse(data) {
  return data?.task === "compare" || Array.isArray(data?.rankings);
}

function syncModeVisibility() {
  const mode = getMode();
  const task = getTask();
  const compareMode = mode === "text" && task === "compare";

  form.dataset.activeTask = compareMode ? "compare" : "predict";
  inputPanel.dataset.mode = mode;
  inputPanel.dataset.task = compareMode ? "compare" : "predict";
  studioGrid.dataset.task = compareMode ? "compare" : "predict";

  if (mode !== "text" && document.querySelector('input[name="task"][value="compare"]')?.checked) {
    document.querySelector('input[name="task"][value="predict"]').checked = true;
    form.dataset.activeTask = "predict";
    inputPanel.dataset.task = "predict";
    studioGrid.dataset.task = "predict";
  }

  if (compareResultsSection) {
    compareResultsSection.hidden = !compareMode;
  }

  if (configChartSection && compareMode) {
    configChartSection.hidden = true;
  }

  if (descriptionLabel) {
    descriptionLabel.textContent = isCompareTask()
      ? "Danh sách laptop cần so sánh"
      : "Mô tả tự nhiên";
  }

  if (descriptionInput) {
    descriptionInput.placeholder = isCompareTask()
      ? example.compareDescription
      : example.description;
    descriptionInput.rows = isCompareTask() ? 12 : 8;
  }

  if (submitButton) {
    submitButton.textContent = isCompareTask() ? "So sánh & xếp hạng" : "Dự đoán giá";
  }
}

function nullableValue(id) {
  const element = document.querySelector(`#${id}`);
  const value = element.value.trim();
  if (id === "condition" && !value) return "unknown";
  if (!value) return null;
  if (["ram_gb", "storage_gb", "screen_size_inch"].includes(id)) return Number(value);
  if (id === "cpu_generation") return Number.parseInt(value, 10);
  return value;
}

function collectRawFeatures() {
  return Object.fromEntries(fields.map((field) => [field, nullableValue(field)]));
}

function setValidation(items, target = "predict") {
  const list = target === "compare" ? compareValidationList : document.querySelector("#validationList");
  if (!list) return;
  list.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
}

function coerceNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatPrice(value) {
  const number = coerceNumber(value);
  if (number === null) return "--";
  const valueInVnd = Math.round(number * 1_000_000);
  return `${valueInVnd.toLocaleString("vi-VN")}<sup>đ</sup>`;
}

function formatPriceRange(range) {
  const low = coerceNumber(range?.low);
  const high = coerceNumber(range?.high);
  if (low === null || high === null) return "--";
  return `${formatPrice(low)} – ${formatPrice(high)}`;
}

function uncertaintyLabel(level) {
  if (level === "low") return "Mức độ không chắc chắn: thấp";
  if (level === "medium") return "Mức độ không chắc chắn: trung bình";
  if (level === "high") return "Mức độ không chắc chắn: cao";
  return "Mức độ không chắc chắn: --";
}

function resetPriceDisplay() {
  predictedPriceRange.textContent = "--";
  predictedPricePoint.textContent = "Giá trung tâm: --";
  completenessBadge.textContent = "--";
  uncertaintyBadge.textContent = "--";
}

function resetCompareDisplay() {
  compareBestPick.textContent = "--";
  compareSummary.textContent = "Chưa có dữ liệu so sánh.";
  compareTableBody.innerHTML = '<tr><td colspan="6">Chưa có kết quả.</td></tr>';
}

function resetConfigChartDisplay() {
  if (configChartState) {
    configChartState.textContent = "Chưa chạy";
    configChartState.classList.remove("is-ready");
  }
  if (window.ConfigChart?.destroy) window.ConfigChart.destroy(configChartPanel);
  if (configChartSection) configChartSection.hidden = true;
}

function resetResultDisplay() {
  resetPriceDisplay();
  resetCompareDisplay();
  resetConfigChartDisplay();
}

function formatGap(gapMillion, gapPct) {
  const gap = coerceNumber(gapMillion);
  if (gap === null) return "--";
  const sign = gap > 0 ? "+" : "";
  const pct = coerceNumber(gapPct);
  const pctText = pct === null ? "" : ` (${sign}${pct}%)`;
  return `${sign}${formatPrice(gap)}${pctText}`;
}

function verdictClass(verdict) {
  if (!verdict) return "";
  if (verdict.includes("Đáng mua")) return "verdict-good";
  if (verdict === "Hợp lý") return "verdict-fair";
  if (verdict.includes("đắt") || verdict.includes("Đắt")) return "verdict-bad";
  return "";
}

function renderCompareResult(data) {
  const rows = data.rankings || [];
  if (!rows.length) {
    resetCompareDisplay();
    return data;
  }

  compareBestPick.textContent = data.best_pick || "--";
  compareSummary.textContent = data.summary || "Chưa có dữ liệu so sánh.";
  compareRunState.textContent = "Hoàn tất";

  compareTableBody.innerHTML = rows
    .map((row) => {
      const verdict = row.verdict || "--";
      return `
        <tr class="${row.rank === 1 ? "compare-top-pick" : ""}">
          <td>#${row.rank}</td>
          <td>${row.label}</td>
          <td>${formatPrice(row.actual_price_million_vnd)}</td>
          <td>${formatPrice(row.predicted_price)}</td>
          <td>${formatGap(row.price_gap_million_vnd, row.price_gap_pct)}</td>
          <td><span class="verdict-pill ${verdictClass(verdict)}">${verdict}</span></td>
        </tr>
      `;
    })
    .join("");

  if (compareResultsSection) {
    compareResultsSection.hidden = false;
    compareResultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  return data;
}

function compareMismatchMessage(data) {
  const messages = [
    "Không nhận được kết quả so sánh. Vui lòng thử lại.",
  ];
  if (data?.predicted_price != null) {
    messages.push(`Giá ước tính: ${formatPrice(data.predicted_price).replace(/<[^>]+>/g, "")}.`);
  }
  return messages;
}

function renderPriceResult(data) {
  const enriched = window.PriceInterval?.enrichPrediction
    ? window.PriceInterval.enrichPrediction(data)
    : data;

  predictedPriceRange.innerHTML = formatPriceRange(enriched.price_range);
  predictedPricePoint.innerHTML = `Giá trung tâm: ${formatPrice(enriched.predicted_price)}`;
  completenessBadge.textContent = `Hoàn thiện input: ${enriched.input_completeness_pct ?? "--"}%`;
  uncertaintyBadge.textContent = uncertaintyLabel(enriched.uncertainty?.level);
  uncertaintyBadge.dataset.level = enriched.uncertainty?.level || "";

  return enriched;
}

async function loadConfigChart(rawFeatures) {
  if (!configChartSection || !configChartPanel || !rawFeatures) return;

  configChartState.textContent = "Đang tính";
  configChartSection.hidden = false;

  try {
    const response = await apiRequest(
      "/api/config-sweep",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ raw_features: rawFeatures }),
      },
      120000,
    );
    const data = await response.json();
    if (!data?.series?.length || !window.ConfigChart?.render) {
      configChartState.textContent = "Không có dữ liệu";
      return;
    }

    window.ConfigChart.render(configChartPanel, data);
    configChartState.textContent = "Sẵn sàng";
    configChartState.classList.add("is-ready");
    configChartSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    configChartState.textContent = "Lỗi";
    if (window.ConfigChart?.destroy) window.ConfigChart.destroy(configChartPanel);
    configChartPanel.innerHTML = `<p class="config-chart-error">${error.message}</p>`;
  }
}

function setLoading(isLoading, task = "predict") {
  submitButton.disabled = isLoading;
  if (task === "compare") {
    compareRunState.textContent = isLoading ? "Đang chạy" : "Hoàn tất";
    return;
  }
  runState.textContent = isLoading ? "Đang chạy" : "Hoàn tất";
}

function fillExample() {
  if (isCompareTask()) {
    descriptionInput.value = example.compareDescription;
    return;
  }

  descriptionInput.value = example.description;
  fields.forEach((field) => {
    const element = document.querySelector(`#${field}`);
    element.value = example.raw_features[field] ?? "";
  });
}

function compareStatus(data) {
  const items = [...(data.validation || [])];
  if (data.summary) items.push(data.summary);
  if (data.best_pick) items.push(`Lựa chọn tốt nhất hiện tại: ${data.best_pick}.`);
  return items;
}

function userFriendlyStatus(data, mode) {
  const items = [];
  items.push(mode === "manual" ? "Đã nhận thông tin từ form." : "Đã phân tích mô tả và trích xuất cấu hình.");
  items.push("Đã kiểm tra các giá trị nhập vào.");
  items.push("Đã chuẩn bị dữ liệu để ước tính giá.");

  if (data.range_source === "client_fallback") {
    items.push("Khoảng giá được tính từ giá trung tâm.");
  }

  if (data.uncertainty?.reason) {
    items.push(data.uncertainty.reason);
  }

  const missing = data.missing_fields || Object.entries(data.raw_features || {})
    .filter(([, value]) => value === null || value === "")
    .map(([key]) => key);

  if (missing.length) {
    items.push(`Thông tin còn thiếu: ${missing.slice(0, 5).join(", ")}${missing.length > 5 ? "..." : ""}.`);
  } else {
    items.push("Thông tin chính đã đủ để tham khảo khoảng giá.");
  }

  return items;
}

async function checkHealth() {
  try {
    const response = await apiRequest("/api/health");
    const data = await response.json();
    const ready = data.ok && data.model_available;
    const staleBackend =
      ready &&
      (data.supports_price_range !== true ||
        data.supports_compare !== true ||
        data.supports_config_sweep !== true);
    setHealth(ready ? (staleBackend ? "Cần làm mới" : "Sẵn sàng") : "Có lỗi");
  } catch (error) {
    setHealth("Không kết nối");
  }
}

async function loadPriceIntervalConfig() {
  if (!window.PriceInterval?.setConfig) return;
  try {
    const response = await apiRequest("/api/price-interval-config");
    const config = await response.json();
    window.PriceInterval.setConfig(config);
  } catch (error) {
    // Keep built-in defaults when config endpoint is unavailable.
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultPanel.classList.remove("is-error");
  if (compareResultsSection) compareResultsSection.classList.remove("is-error");

  const task = form.dataset.activeTask === "compare" || isCompareTask() ? "compare" : "predict";
  setLoading(true, task);
  setValidation(task === "compare" ? ["Đang phân tích và so sánh từng laptop..."] : ["Đang xử lý yêu cầu..."], task);
  resetResultDisplay();

  const mode = getMode();
  const description = descriptionInput.value.trim();
  const endpoint = task === "compare" ? "/api/compare" : "/api/predict";
  const payload =
    task === "compare"
      ? { description }
      : mode === "manual"
        ? { mode, raw_features: collectRawFeatures() }
        : { mode, task: "predict", description };

  try {
    const response = await apiRequest(
      endpoint,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      },
      task === "compare" ? 120000 : 60000,
    );
    const data = await response.json();

    if (task === "compare" || isCompareResponse(data)) {
      if (!isCompareResponse(data)) {
        if (compareResultsSection) compareResultsSection.classList.add("is-error");
        setValidation(compareMismatchMessage(data), "compare");
        compareRunState.textContent = "Lỗi";
        return;
      }

      const compared = renderCompareResult(data);
      setValidation(compareStatus(compared), "compare");
      return;
    }

    runState.textContent = "Hoàn tất";
    const enriched = renderPriceResult(data);
    setValidation(userFriendlyStatus(enriched, mode), "predict");

    fields.forEach((field) => {
      const element = document.querySelector(`#${field}`);
      if (element && enriched.raw_features && enriched.raw_features[field] !== null) {
        element.value = enriched.raw_features[field];
      }
    });

    await loadConfigChart(enriched.raw_features || data.raw_features);
  } catch (error) {
    if (task === "compare") {
      if (compareResultsSection) compareResultsSection.classList.add("is-error");
      compareRunState.textContent = "Lỗi";
      setValidation([error.message], "compare");
    } else {
      resultPanel.classList.add("is-error");
      runState.textContent = "Lỗi";
      setValidation([error.message], "predict");
    }
  } finally {
    submitButton.disabled = false;
  }
});

document.querySelector("#fillExample").addEventListener("click", fillExample);

document.querySelectorAll('input[name="mode"], input[name="task"]').forEach((input) => {
  input.addEventListener("change", () => {
    if (input.name === "task") {
      descriptionInput.value = "";
    }
    syncModeVisibility();
    form.dataset.activeTask = getTask();
  });
});

document.querySelector("#clearForm").addEventListener("click", () => {
  descriptionInput.value = "";
  fields.forEach((field) => {
    document.querySelector(`#${field}`).value = "";
  });
  resetResultDisplay();
  runState.textContent = "Chưa chạy";
  compareRunState.textContent = "Chưa chạy";
  setValidation(["Chờ dữ liệu đầu vào."], "predict");
  setValidation(["Chờ dữ liệu so sánh."], "compare");
});

document.querySelectorAll(".accordion-item").forEach((item) => {
  item.addEventListener("mouseenter", () => {
    document.querySelectorAll(".accordion-item").forEach((entry) => entry.classList.remove("active"));
    item.classList.add("active");
  });
});

fillExample();
syncModeVisibility();
loadPriceIntervalConfig();
checkHealth();

if (window.gsap && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  gsap.from(".hero-topline", {
    y: -20,
    opacity: 0,
    duration: 0.55,
    ease: "power3.out",
  });

  gsap.from(".hero-copy > *", {
    y: 24,
    opacity: 0,
    duration: 0.55,
    stagger: 0.045,
    ease: "power3.out",
  });

  gsap.from(".input-panel, .result-panel", {
    y: 24,
    opacity: 0,
    duration: 0.55,
    stagger: 0.05,
    ease: "power3.out",
  });

  if (window.ScrollTrigger) {
    gsap.registerPlugin(ScrollTrigger);
    gsap.utils.toArray(".process-card, .accordion-item").forEach((element) => {
      gsap.from(element, {
        y: 28,
        opacity: 0,
        duration: 0.45,
        ease: "power2.out",
        scrollTrigger: {
          trigger: element,
          start: "top 92%",
          once: true,
        },
      });
    });
  }
}
