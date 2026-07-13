(function initConfigChart(global) {
  const DEFAULT_GRAY = "rgba(248, 243, 232, 0.34)";
  const DEFAULT_BAND = "rgba(248, 243, 232, 0.08)";
  const INACTIVE_GRAY = "rgba(248, 243, 232, 0.18)";
  const INACTIVE_BAND = "rgba(248, 243, 232, 0.04)";
  const MUTED = "rgba(248, 243, 232, 0.62)";
  const GRID = "rgba(248, 243, 232, 0.1)";

  function coerceNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function formatAxisPrice(million) {
    const valueInVnd = Math.round(million * 1_000_000);
    if (valueInVnd >= 1_000_000_000) {
      return `${(valueInVnd / 1_000_000_000).toFixed(1)} tỷ`;
    }
    return `${Math.round(valueInVnd / 1_000_000)} tr`;
  }

  function formatTooltipPrice(million) {
    const valueInVnd = Math.round(million * 1_000_000);
    return `${valueInVnd.toLocaleString("vi-VN")} đ`;
  }

  function createSvgEl(name, attrs = {}) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
    return el;
  }

  function scaleLinear(domain, range) {
    const [d0, d1] = domain;
    const [r0, r1] = range;
    const span = d1 - d0 || 1;
    return (value) => r0 + ((value - d0) / span) * (r1 - r0);
  }

  function destroy(container) {
    if (!container) return;
    const tooltip = container._configChartTooltip;
    if (tooltip?.parentNode) {
      tooltip.parentNode.removeChild(tooltip);
    }
    container.innerHTML = "";
    delete container.dataset.configChartBound;
    delete container._configChartTooltip;
  }

  function positionTooltip(tooltip, clientX, clientY) {
    const offset = 16;
    const padding = 12;
    tooltip.hidden = false;
    tooltip.style.visibility = "hidden";
    tooltip.style.left = "0px";
    tooltip.style.top = "0px";

    const rect = tooltip.getBoundingClientRect();
    let left = clientX + offset;
    let top = clientY - rect.height - offset;

    if (left + rect.width > window.innerWidth - padding) {
      left = clientX - rect.width - offset;
    }
    if (left < padding) left = padding;
    if (top < padding) top = clientY + offset;
    if (top + rect.height > window.innerHeight - padding) {
      top = window.innerHeight - rect.height - padding;
    }

    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
    tooltip.style.visibility = "visible";
  }

  function render(container, payload, options = {}) {
    if (!container || !payload?.series?.length) {
      destroy(container);
      return null;
    }

    destroy(container);
    container.dataset.configChartBound = "true";

    const width = options.width || container.clientWidth || 1100;
    const height = options.height || Math.max(620, Math.round(width * 0.52));
    const margin = { top: 28, right: 32, bottom: 64, left: 88 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    const ramValues = payload.ram_values || [];
    const series = payload.series;

    let yMin = Infinity;
    let yMax = -Infinity;
    series.forEach((entry) => {
      entry.points.forEach((point) => {
        yMin = Math.min(yMin, point.price_range.low, point.price_mean);
        yMax = Math.max(yMax, point.price_range.high, point.price_mean);
      });
    });
    if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) return null;
    const yPadding = Math.max((yMax - yMin) * 0.08, 0.5);
    yMin = Math.max(0, yMin - yPadding);
    yMax += yPadding;

    const xStep = ramValues.length > 1 ? plotWidth / (ramValues.length - 1) : plotWidth;
    const xAt = (index) => margin.left + index * xStep;
    const yAt = scaleLinear([yMin, yMax], [margin.top + plotHeight, margin.top]);

    const root = document.createElement("div");
    root.className = "config-chart-root";

    const svg = createSvgEl("svg", {
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": "Biểu đồ giá dự đoán theo RAM và dung lượng lưu trữ",
    });

    const defs = createSvgEl("defs");
    series.forEach((entry, index) => {
      const glow = createSvgEl("filter", {
        id: `config-chart-glow-${index}`,
        x: "-30%",
        y: "-30%",
        width: "160%",
        height: "160%",
      });
      const blur = createSvgEl("feGaussianBlur", { stdDeviation: "3", result: "blur" });
      const merge = createSvgEl("feMerge");
      const mergeNode1 = createSvgEl("feMergeNode", { in: "blur" });
      const mergeNode2 = createSvgEl("feMergeNode", { in: "SourceGraphic" });
      merge.appendChild(mergeNode1);
      merge.appendChild(mergeNode2);
      glow.appendChild(blur);
      glow.appendChild(merge);
      defs.appendChild(glow);
    });
    svg.appendChild(defs);

    const gridGroup = createSvgEl("g");
    const ticks = 5;
    for (let index = 0; index <= ticks; index += 1) {
      const value = yMin + ((yMax - yMin) * index) / ticks;
      const y = yAt(value);
      gridGroup.appendChild(
        createSvgEl("line", {
          x1: margin.left,
          x2: margin.left + plotWidth,
          y1: y,
          y2: y,
          stroke: GRID,
        }),
      );
      const label = createSvgEl("text", {
        x: margin.left - 10,
        y: y + 4,
        fill: MUTED,
        "text-anchor": "end",
        "font-size": "13",
      });
      label.textContent = formatAxisPrice(value);
      gridGroup.appendChild(label);
    }
    svg.appendChild(gridGroup);

    ramValues.forEach((ram, index) => {
      const x = xAt(index);
      const tick = createSvgEl("text", {
        x,
        y: height - 22,
        fill: MUTED,
        "text-anchor": "middle",
        "font-size": "13",
        "font-weight": "600",
      });
      tick.textContent = `${ram}GB`;
      svg.appendChild(tick);
    });

    const xLabel = createSvgEl("text", {
      x: margin.left + plotWidth / 2,
      y: height - 2,
      fill: MUTED,
      "text-anchor": "middle",
      "font-size": "14",
      "font-weight": "700",
    });
    xLabel.textContent = "RAM";
    svg.appendChild(xLabel);

    const yLabel = createSvgEl("text", {
      x: 16,
      y: margin.top + plotHeight / 2,
      fill: MUTED,
      "text-anchor": "middle",
      "font-size": "14",
      "font-weight": "700",
      transform: `rotate(-90 20 ${margin.top + plotHeight / 2})`,
    });
    yLabel.textContent = "Giá dự đoán (VND)";
    svg.appendChild(yLabel);

    const plotGroup = createSvgEl("g");
    const lineGroup = createSvgEl("g");
    const markerGroup = createSvgEl("g");
    const hitGroup = createSvgEl("g");
    const state = { activeIndex: null, pinnedIndex: null, activePointIndex: null };

    function bringSeriesToFront(group, activeIndex) {
      if (activeIndex === null) return;
      const nodes = [...group.querySelectorAll(`[data-series-index="${activeIndex}"]`)];
      nodes.forEach((node) => group.appendChild(node));
    }

    function bringPointToFront(seriesIndex, pointIndex) {
      if (seriesIndex === null || pointIndex === null) return;
      const band = plotGroup.querySelector(
        `[data-chart-part="band"][data-series-index="${seriesIndex}"][data-point-index="${pointIndex}"]`,
      );
      if (band) plotGroup.appendChild(band);
    }

    function applyHighlight() {
      const active = state.pinnedIndex ?? state.activeIndex;
      const activePoint = state.activePointIndex;
      const hasFocus = active !== null;
      const hasPointFocus = hasFocus && activePoint !== null;

      plotGroup.querySelectorAll('[data-chart-part="band"]').forEach((node) => {
        const seriesIndex = Number(node.dataset.seriesIndex);
        const pointIndex = Number(node.dataset.pointIndex);
        const isSeriesActive = hasFocus && active === seriesIndex;
        const isPointActive = hasPointFocus && isSeriesActive && activePoint === pointIndex;

        if (isPointActive) {
          node.setAttribute("fill", `${series[seriesIndex].color}88`);
          node.setAttribute("stroke", series[seriesIndex].color);
          node.setAttribute("stroke-width", "2.5");
        } else if (isSeriesActive) {
          node.setAttribute("fill", `${series[seriesIndex].color}33`);
          node.removeAttribute("stroke");
        } else if (hasFocus) {
          node.setAttribute("fill", INACTIVE_BAND);
          node.removeAttribute("stroke");
        } else {
          node.setAttribute("fill", DEFAULT_BAND);
          node.removeAttribute("stroke");
        }
      });

      lineGroup.querySelectorAll("[data-series-index]").forEach((node) => {
        const index = Number(node.dataset.seriesIndex);
        const isActive = hasFocus && active === index;
        const color = isActive ? series[index].color : hasFocus ? INACTIVE_GRAY : DEFAULT_GRAY;

        node.setAttribute("stroke", color);
        node.setAttribute("stroke-width", isActive ? "5" : hasFocus ? "1.4" : "2.2");
        if (isActive) {
          node.setAttribute("filter", `url(#config-chart-glow-${index})`);
        } else {
          node.removeAttribute("filter");
        }
      });

      markerGroup.querySelectorAll("[data-series-index]").forEach((node) => {
        const seriesIndex = Number(node.dataset.seriesIndex);
        const pointIndex = Number(node.dataset.pointIndex);
        const isSeriesActive = hasFocus && active === seriesIndex;
        const isPointActive = hasPointFocus && isSeriesActive && activePoint === pointIndex;

        node.setAttribute("fill", isPointActive ? series[seriesIndex].color : "transparent");
        node.setAttribute("stroke", isPointActive ? "#090b10" : "transparent");
        node.setAttribute("stroke-width", isPointActive ? "2.5" : "0");
        node.setAttribute("r", isPointActive ? "7" : "0");
      });

      bringSeriesToFront(plotGroup, active);
      bringSeriesToFront(lineGroup, active);
      bringSeriesToFront(markerGroup, active);
      if (hasPointFocus) {
        bringPointToFront(active, activePoint);
      }

      root.querySelectorAll(".config-chart-legend-item").forEach((button, index) => {
        const isActive = active === index;
        button.classList.toggle("is-active", isActive);
        const swatch = button.querySelector("span");
        if (swatch) {
          swatch.style.background = isActive ? series[index].color : "rgba(248, 243, 232, 0.35)";
          swatch.style.boxShadow = isActive ? `0 0 0 3px ${series[index].color}44` : "none";
        }
      });
    }

    function setActive(index) {
      state.pinnedIndex = state.pinnedIndex === index ? null : index;
      state.activeIndex = state.pinnedIndex;
      state.activePointIndex = null;
      applyHighlight();
    }

    function nearestSeriesAtPoint(ramIndex, relativeY) {
      let nearestSeries = 0;
      let minDistance = Infinity;

      series.forEach((entry, seriesIndex) => {
        const point = entry.points[ramIndex];
        if (!point) return;
        const pointY = yAt(point.price_mean);
        const distance = Math.abs(pointY - relativeY);
        if (distance < minDistance) {
          minDistance = distance;
          nearestSeries = seriesIndex;
        }
      });

      return nearestSeries;
    }

    series.forEach((entry, seriesIndex) => {
      const bandWidth = Math.min(42, xStep * 0.55);
      entry.points.forEach((point, pointIndex) => {
        const x = xAt(pointIndex);
        const lowY = yAt(point.price_range.low);
        const highY = yAt(point.price_range.high);
        const rect = createSvgEl("rect", {
          x: x - bandWidth / 2,
          y: Math.min(lowY, highY),
          width: bandWidth,
          height: Math.abs(highY - lowY) || 1,
          rx: 4,
          fill: DEFAULT_BAND,
          "data-series-index": seriesIndex,
          "data-point-index": pointIndex,
          "data-chart-part": "band",
        });
        plotGroup.appendChild(rect);
      });

      const path = entry.points
        .map((point, pointIndex) => {
          const x = xAt(pointIndex);
          const y = yAt(point.price_mean);
          return `${pointIndex === 0 ? "M" : "L"} ${x} ${y}`;
        })
        .join(" ");

      const line = createSvgEl("path", {
        d: path,
        fill: "none",
        stroke: DEFAULT_GRAY,
        "stroke-width": "2.2",
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
        "data-series-index": seriesIndex,
        "data-chart-part": "line",
      });
      lineGroup.appendChild(line);

      entry.points.forEach((point, pointIndex) => {
        const marker = createSvgEl("circle", {
          cx: xAt(pointIndex),
          cy: yAt(point.price_mean),
          r: 0,
          fill: "transparent",
          stroke: "transparent",
          "stroke-width": "2",
          "data-series-index": seriesIndex,
          "data-point-index": pointIndex,
          "data-chart-part": "marker",
        });
        markerGroup.appendChild(marker);
      });

      const hitLine = createSvgEl("path", {
        d: path,
        fill: "none",
        stroke: "transparent",
        "stroke-width": "22",
        "data-series-index": seriesIndex,
        "data-chart-part": "hit",
      });
      hitLine.addEventListener("mouseenter", () => {
        if (state.pinnedIndex === null) {
          state.activeIndex = seriesIndex;
          state.activePointIndex = null;
          applyHighlight();
        }
      });
      hitLine.addEventListener("mouseleave", () => {
        if (state.pinnedIndex === null) {
          state.activeIndex = null;
          state.activePointIndex = null;
          applyHighlight();
        }
      });
      hitLine.addEventListener("click", () => setActive(seriesIndex));
      hitGroup.appendChild(hitLine);
    });

    svg.appendChild(plotGroup);
    svg.appendChild(lineGroup);
    svg.appendChild(markerGroup);
    svg.appendChild(hitGroup);

    const tooltip = document.createElement("div");
    tooltip.className = "config-chart-tooltip";
    tooltip.hidden = true;
    document.body.appendChild(tooltip);
    container._configChartTooltip = tooltip;

    svg.addEventListener("mousemove", (event) => {
      const bounds = svg.getBoundingClientRect();
      const relativeX = ((event.clientX - bounds.left) / bounds.width) * width;
      const relativeY = ((event.clientY - bounds.top) / bounds.height) * height;
      const ramIndex = Math.round((relativeX - margin.left) / xStep);
      if (ramIndex < 0 || ramIndex >= ramValues.length) {
        tooltip.hidden = true;
        state.activePointIndex = null;
        applyHighlight();
        return;
      }

      const nearestSeries =
        state.pinnedIndex ?? nearestSeriesAtPoint(ramIndex, relativeY);
      state.activePointIndex = ramIndex;
      if (state.pinnedIndex === null) {
        state.activeIndex = nearestSeries;
      }
      applyHighlight();

      const point = series[nearestSeries]?.points?.[ramIndex];
      if (!point) {
        tooltip.hidden = true;
        return;
      }

      tooltip.innerHTML = `
        <strong>RAM ${ramValues[ramIndex]}GB · ${series[nearestSeries].storage_label}</strong>
        <span>Giá trung bình: ${formatTooltipPrice(point.price_mean)}</span>
        <span>Khoảng: ${formatTooltipPrice(point.price_range.low)} – ${formatTooltipPrice(point.price_range.high)}</span>
      `;
      positionTooltip(tooltip, event.clientX, event.clientY);
    });

    svg.addEventListener("mouseleave", () => {
      tooltip.hidden = true;
      state.activePointIndex = null;
      if (state.pinnedIndex === null) {
        state.activeIndex = null;
      }
      applyHighlight();
    });

    const legend = document.createElement("div");
    legend.className = "config-chart-legend";
    series.forEach((entry, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "config-chart-legend-item";
      button.innerHTML = `<span style="background:${entry.color}"></span>${entry.storage_label}`;
      button.addEventListener("mouseenter", () => {
        if (state.pinnedIndex === null) {
          state.activeIndex = index;
          state.activePointIndex = null;
          applyHighlight();
        }
      });
      button.addEventListener("mouseleave", () => {
        if (state.pinnedIndex === null) {
          state.activeIndex = null;
          state.activePointIndex = null;
          applyHighlight();
        }
      });
      button.addEventListener("click", () => setActive(index));
      legend.appendChild(button);
    });

    const chartWrap = document.createElement("div");
    chartWrap.className = "config-chart-canvas";
    chartWrap.appendChild(svg);

    root.appendChild(chartWrap);
    root.appendChild(legend);
    container.appendChild(root);
    applyHighlight();

    return { destroy: () => destroy(container) };
  }

  global.ConfigChart = {
    render,
    destroy,
    formatTooltipPrice,
    coerceNumber,
  };
})(window);
