(function initPriceInterval(global) {
  const DEFAULT_CONFIG = {
    raw_laptop_fields: [
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
    ],
    critical_fields: ["brand", "model", "ram_gb", "storage_gb", "cpu_text", "gpu_text"],
    segment_rmse: {
      Q1_low: 2.805,
      Q2: 2.575,
      Q3: 3.927,
      Q4: 4.234,
      Q5_high: 11.268,
    },
    segment_upper_bounds: [4.725, 8.407, 13.576, 27.362],
    missing_field_weights: {
      brand: 0.18,
      model: 0.14,
      ram_gb: 0.16,
      storage_gb: 0.14,
      cpu_text: 0.16,
      gpu_text: 0.12,
      storage_type: 0.06,
      screen_size_inch: 0.05,
      condition: 0.04,
      warranty_status: 0.03,
      cpu_brand: 0.03,
      cpu_family: 0.03,
      cpu_generation: 0.02,
      cpu_suffix: 0.02,
    },
    no_info_flag_weight: 0.04,
    max_uncertainty_multiplier: 2.6,
    min_price_million: 0.5,
  };

  let config = DEFAULT_CONFIG;

  function setConfig(nextConfig) {
    if (nextConfig && typeof nextConfig === "object") {
      config = { ...DEFAULT_CONFIG, ...nextConfig };
    }
  }

  function isMissing(value) {
    if (value === null || value === undefined) return true;
    if (typeof value === "string") return !value.trim();
    if (typeof value === "number" && Number.isNaN(value)) return true;
    return false;
  }

  function priceSegment(predictedPriceMillion) {
    const price = Math.max(Number(predictedPriceMillion) || 0, 0);
    const bounds = config.segment_upper_bounds;
    if (price <= bounds[0]) return "Q1_low";
    if (price <= bounds[1]) return "Q2";
    if (price <= bounds[2]) return "Q3";
    if (price <= bounds[3]) return "Q4";
    return "Q5_high";
  }

  function missingFields(rawFeatures) {
    return config.raw_laptop_fields.filter((field) => isMissing(rawFeatures?.[field]));
  }

  function completenessPct(rawFeatures) {
    const missingCount = missingFields(rawFeatures).length;
    const present = config.raw_laptop_fields.length - missingCount;
    return Math.round((100 * present) / config.raw_laptop_fields.length * 10) / 10;
  }

  function uncertaintyMultiplier(rawFeatures, encodedFeatures) {
    let multiplier = 1;
    for (const [field, weight] of Object.entries(config.missing_field_weights)) {
      if (isMissing(rawFeatures?.[field])) multiplier += weight;
    }

    if (encodedFeatures) {
      const noInfoFlags = Object.entries(encodedFeatures).filter(
        ([key, value]) => key.startsWith("no_info_") && Number(value) === 1,
      ).length;
      multiplier += noInfoFlags * config.no_info_flag_weight;
    }

    return Math.min(multiplier, config.max_uncertainty_multiplier);
  }

  function uncertaintyLevel(multiplier, completeness) {
    if (multiplier <= 1.2 && completeness >= 80) return "low";
    if (multiplier <= 1.7 && completeness >= 55) return "medium";
    return "high";
  }

  function uncertaintyReason(level, missing) {
    if (!missing.length) {
      return "Đủ thông tin chính; khoảng giá dựa trên độ chính xác theo phân khúc giá.";
    }

    const criticalMissing = missing.filter((field) => config.critical_fields.includes(field));
    if (level === "high") {
      if (criticalMissing.length) {
        return `Thiếu nhiều thông tin quan trọng (${criticalMissing.slice(0, 4).join(", ")}); khoảng giá được nới rộng.`;
      }
      return "Một số thông tin còn thiếu; khoảng giá được nới rộng để phản ánh độ không chắc chắn.";
    }

    if (criticalMissing.length) {
      return `Thiếu ${criticalMissing.slice(0, 3).join(", ")}; khoảng giá rộng hơn mức trung bình.`;
    }
    return "Thiếu vài thông tin phụ; khoảng giá điều chỉnh nhẹ theo mức độ hoàn thiện input.";
  }

  function buildPricePrediction(predictedPriceMillion, rawFeatures, encodedFeatures) {
    const point = Math.round(Number(predictedPriceMillion) * 1000) / 1000;
    const segment = priceSegment(point);
    const baseMargin = config.segment_rmse[segment];
    const multiplier = uncertaintyMultiplier(rawFeatures || {}, encodedFeatures || null);
    const halfWidth = baseMargin * multiplier;
    const low = Math.max(config.min_price_million, point - halfWidth);
    const high = point + halfWidth;
    const completeness = completenessPct(rawFeatures || {});
    const missing = missingFields(rawFeatures || {});
    const level = uncertaintyLevel(multiplier, completeness);

    return {
      predicted_price: point,
      price_unit: "million_vnd",
      price_range: {
        low: Math.round(low * 100) / 100,
        high: Math.round(high * 100) / 100,
        half_width_million_vnd: Math.round(halfWidth * 100) / 100,
      },
      price_segment: segment,
      input_completeness_pct: completeness,
      missing_fields: missing,
      uncertainty: {
        level,
        multiplier: Math.round(multiplier * 100) / 100,
        base_rmse_million_vnd: baseMargin,
        reason: uncertaintyReason(level, missing),
      },
    };
  }

  function hasCompleteRange(payload) {
    const low = Number(payload?.price_range?.low);
    const high = Number(payload?.price_range?.high);
    return Number.isFinite(low) && Number.isFinite(high);
  }

  function enrichPrediction(payload) {
    const point = Number(payload?.predicted_price);
    if (!Number.isFinite(point)) return payload;

    if (
      hasCompleteRange(payload) &&
      payload.input_completeness_pct != null &&
      payload.uncertainty?.level
    ) {
      return payload;
    }

    const enriched = buildPricePrediction(
      point,
      payload.raw_features || {},
      payload.encoded_features || null,
    );

    return {
      ...payload,
      ...enriched,
      range_source: hasCompleteRange(payload) ? "server" : "client_fallback",
    };
  }

  global.PriceInterval = {
    setConfig,
    buildPricePrediction,
    enrichPrediction,
    hasCompleteRange,
  };
})(window);
