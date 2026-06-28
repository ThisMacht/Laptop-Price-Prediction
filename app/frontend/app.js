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
const predictedPrice = document.querySelector("#predictedPrice");
const validationList = document.querySelector("#validationList");
const healthStatus = document.querySelector("#healthStatus");
const inputPanel = document.querySelector(".input-panel");
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

async function apiRequest(path, options = {}) {
  const bases = activeApiBase === null ? apiBaseCandidates() : [activeApiBase, ...apiBaseCandidates()];
  const errors = [];

  for (const base of [...new Set(bases)]) {
    try {
      const response = await fetchWithTimeout(`${base}${path}`, options);
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

function syncModeVisibility() {
  inputPanel.dataset.mode = getMode();
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

function setValidation(items) {
  validationList.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    validationList.appendChild(li);
  });
}

function formatPrice(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "--";
  const valueInVnd = Math.round(value * 1_000_000);
  return `${valueInVnd.toLocaleString("vi-VN")}<sup>đ</sup>`;
}

function setLoading(isLoading) {
  form.querySelector(".primary-button").disabled = isLoading;
  runState.textContent = isLoading ? "Đang chạy" : "Hoàn tất";
}

function fillExample() {
  document.querySelector("#description").value = example.description;
  fields.forEach((field) => {
    const element = document.querySelector(`#${field}`);
    element.value = example.raw_features[field] ?? "";
  });
}

function userFriendlyStatus(data, mode) {
  const items = [];
  items.push(mode === "manual" ? "Đã nhận thông tin từ form." : "Đã đọc mô tả và rút thông tin cấu hình.");
  items.push("Đã kiểm tra các giá trị nhập vào.");
  items.push("Đã chuẩn hóa cấu hình cho model dự đoán.");

  const missing = Object.entries(data.raw_features || {})
    .filter(([, value]) => value === null || value === "")
    .map(([key]) => key);

  if (missing.length) {
    items.push(`Một số thông tin còn thiếu, kết quả có thể kém chính xác hơn: ${missing.slice(0, 4).join(", ")}${missing.length > 4 ? "..." : ""}.`);
  } else {
    items.push("Thông tin chính đã đủ để tham khảo giá.");
  }

  return items;
}

async function checkHealth() {
  try {
    const response = await apiRequest("/api/health");
    const data = await response.json();
    setHealth(data.ok && data.model_available ? "Sẵn sàng" : "Có lỗi");
  } catch (error) {
    setHealth("Không kết nối");
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  document.querySelector(".result-panel").classList.remove("is-error");
  setLoading(true);
  setValidation(["Đang gửi dữ liệu tới backend."]);
  predictedPrice.textContent = "--";

  const mode = getMode();
  const payload =
    mode === "manual"
      ? { mode, raw_features: collectRawFeatures() }
      : { mode, description: document.querySelector("#description").value.trim() };

  try {
    const response = await apiRequest("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();

    runState.textContent = "Hoàn tất";
    predictedPrice.innerHTML = formatPrice(data.predicted_price);
    setValidation(userFriendlyStatus(data, mode));

    fields.forEach((field) => {
      const element = document.querySelector(`#${field}`);
      if (element && data.raw_features && data.raw_features[field] !== null) {
        element.value = data.raw_features[field];
      }
    });
  } catch (error) {
    document.querySelector(".result-panel").classList.add("is-error");
    runState.textContent = "Lỗi";
    setValidation([error.message]);
  } finally {
    form.querySelector(".primary-button").disabled = false;
  }
});

document.querySelector("#fillExample").addEventListener("click", fillExample);

document.querySelectorAll('input[name="mode"]').forEach((input) => {
  input.addEventListener("change", syncModeVisibility);
});

document.querySelector("#clearForm").addEventListener("click", () => {
  document.querySelector("#description").value = "";
  fields.forEach((field) => {
    document.querySelector(`#${field}`).value = "";
  });
  predictedPrice.textContent = "--";
  runState.textContent = "Chưa chạy";
  setValidation(["Chờ dữ liệu đầu vào."]);
});

document.querySelectorAll(".accordion-item").forEach((item) => {
  item.addEventListener("mouseenter", () => {
    document.querySelectorAll(".accordion-item").forEach((entry) => entry.classList.remove("active"));
    item.classList.add("active");
  });
});

fillExample();
syncModeVisibility();
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
