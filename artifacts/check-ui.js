const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
  await page.goto('http://127.0.0.1:8000/', { waitUntil: 'networkidle' });
  await page.screenshot({ path: 'artifacts/ui-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 1100 });
  await page.goto('http://127.0.0.1:8000/', { waitUntil: 'networkidle' });
  await page.screenshot({ path: 'artifacts/ui-mobile.png', fullPage: true });
  await browser.close();
})();
