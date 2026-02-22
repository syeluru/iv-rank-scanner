// Schwab Trade Confirm - Content Script
// Intercepts all clickable elements and requires confirmation

(function () {
  "use strict";

  let enabled = true;
  let overlay = null;
  let pendingEvent = null;
  let pendingTarget = null;
  let confirmTimer = null;

  // Load enabled state from storage
  chrome.storage.local.get("schwabConfirmEnabled", (data) => {
    enabled = data.schwabConfirmEnabled !== false; // default true
  });

  // Listen for toggle messages from popup
  chrome.storage.onChanged.addListener((changes) => {
    if (changes.schwabConfirmEnabled) {
      enabled = changes.schwabConfirmEnabled.newValue;
    }
  });

  // Elements that should trigger confirmation
  function isClickable(el) {
    if (!el) return false;

    const tag = el.tagName?.toLowerCase();

    // Direct clickable elements
    if (tag === "button") return true;
    if (tag === "a" && el.href) return true;
    if (tag === "input" && ["submit", "button", "reset"].includes(el.type?.toLowerCase())) return true;

    // Elements with click roles
    const role = el.getAttribute("role");
    if (["button", "link", "menuitem", "tab", "option"].includes(role)) return true;

    // Elements with onclick handlers or cursor pointer
    if (el.onclick) return true;

    // Check computed style for pointer cursor (indicates clickable)
    const style = window.getComputedStyle(el);
    if (style.cursor === "pointer") return true;

    return false;
  }

  // Walk up the DOM to find the actual clickable element
  function findClickableAncestor(el, maxDepth = 5) {
    let current = el;
    let depth = 0;
    while (current && depth < maxDepth) {
      if (isClickable(current)) return current;
      current = current.parentElement;
      depth++;
    }
    return null;
  }

  // Get a human-readable description of what's being clicked
  function describeElement(el) {
    const tag = el.tagName?.toLowerCase();

    // Get visible text
    let text = el.innerText?.trim() || el.textContent?.trim() || "";
    if (text.length > 100) text = text.substring(0, 100) + "...";

    // Get aria-label
    const ariaLabel = el.getAttribute("aria-label");

    // Get title
    const title = el.getAttribute("title");

    // Get value (for inputs)
    const value = el.value;

    // Get placeholder
    const placeholder = el.getAttribute("placeholder");

    // Build description
    let desc = "";

    if (text) {
      desc = text;
    } else if (ariaLabel) {
      desc = ariaLabel;
    } else if (title) {
      desc = title;
    } else if (value) {
      desc = value;
    } else if (placeholder) {
      desc = placeholder;
    } else {
      desc = `<${tag}> element`;
    }

    // Add context about what type of element
    let typeHint = "";
    if (tag === "a") typeHint = "Link";
    else if (tag === "button") typeHint = "Button";
    else if (tag === "input") typeHint = "Input";
    else typeHint = "Element";

    return { text: desc, type: typeHint };
  }

  // Check if this click is on our own overlay (should not be intercepted)
  function isOverlayClick(el) {
    if (!overlay) return false;
    return overlay.contains(el);
  }

  // Determine danger level based on button text
  function getDangerLevel(text) {
    const lower = text.toLowerCase();

    // High danger - trade execution
    const highDanger = [
      "place order", "submit order", "confirm", "execute",
      "buy", "sell", "trade", "review order", "send order",
      "place trade", "submit trade", "market order", "limit order",
      "close position", "cancel order", "modify order", "replace order"
    ];
    if (highDanger.some((kw) => lower.includes(kw))) return "high";

    // Medium danger - navigation that could lead to trades
    const medDanger = [
      "next", "continue", "proceed", "accept", "agree",
      "ok", "yes", "delete", "remove", "transfer"
    ];
    if (medDanger.some((kw) => lower.includes(kw))) return "medium";

    return "low";
  }

  // Create and show the confirmation overlay
  function showConfirmation(target, originalEvent) {
    if (overlay) removeOverlay();

    const info = describeElement(target);
    const danger = getDangerLevel(info.text);

    overlay = document.createElement("div");
    overlay.id = "schwab-confirm-overlay";

    const dangerClass = `schwab-confirm-danger-${danger}`;
    const dangerLabel =
      danger === "high" ? "HIGH RISK ACTION" :
      danger === "medium" ? "CAUTION" :
      "CONFIRM ACTION";

    const delayMs = danger === "high" ? 2000 : danger === "medium" ? 1000 : 500;

    overlay.innerHTML = `
      <div class="schwab-confirm-modal ${dangerClass}">
        <div class="schwab-confirm-header">
          <span class="schwab-confirm-icon">${danger === "high" ? "&#9888;" : danger === "medium" ? "&#9888;" : "&#10067;"}</span>
          <span class="schwab-confirm-title">${dangerLabel}</span>
        </div>
        <div class="schwab-confirm-body">
          <div class="schwab-confirm-label">You are about to click:</div>
          <div class="schwab-confirm-target">
            <span class="schwab-confirm-type">${info.type}</span>
            <span class="schwab-confirm-text">"${info.text.replace(/"/g, '&quot;').replace(/</g, '&lt;')}"</span>
          </div>
          ${danger === "high" ? '<div class="schwab-confirm-warning">This looks like a trade execution. Are you ABSOLUTELY sure?</div>' : ""}
        </div>
        <div class="schwab-confirm-footer">
          <button class="schwab-confirm-cancel" id="schwab-cancel-btn">Cancel</button>
          <button class="schwab-confirm-ok ${dangerClass}" id="schwab-ok-btn" disabled>
            ${danger === "high" ? "Yes, Execute" : "Yes, Click It"}
            <span class="schwab-confirm-countdown" id="schwab-countdown"></span>
          </button>
        </div>
      </div>
    `;

    document.body.appendChild(overlay);

    // Highlight the target element
    target.classList.add("schwab-confirm-highlight");

    const cancelBtn = document.getElementById("schwab-cancel-btn");
    const okBtn = document.getElementById("schwab-ok-btn");
    const countdown = document.getElementById("schwab-countdown");

    // Countdown before confirm button activates
    let remaining = delayMs / 1000;
    countdown.textContent = ` (${remaining.toFixed(1)}s)`;

    const countdownInterval = setInterval(() => {
      remaining -= 0.1;
      if (remaining <= 0) {
        clearInterval(countdownInterval);
        countdown.textContent = "";
        okBtn.disabled = false;
        okBtn.classList.add("schwab-confirm-ready");
      } else {
        countdown.textContent = ` (${remaining.toFixed(1)}s)`;
      }
    }, 100);

    // Cancel
    cancelBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      e.preventDefault();
      clearInterval(countdownInterval);
      removeOverlay();
      target.classList.remove("schwab-confirm-highlight");
    });

    // Confirm - re-dispatch the original click
    okBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      e.preventDefault();
      clearInterval(countdownInterval);
      removeOverlay();
      target.classList.remove("schwab-confirm-highlight");

      // Temporarily disable interception and re-click
      pendingTarget = target;
      target.click();
      pendingTarget = null;
    });

    // Click outside modal = cancel
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) {
        clearInterval(countdownInterval);
        removeOverlay();
        target.classList.remove("schwab-confirm-highlight");
      }
    });

    // Escape key = cancel
    const escHandler = (e) => {
      if (e.key === "Escape") {
        clearInterval(countdownInterval);
        removeOverlay();
        target.classList.remove("schwab-confirm-highlight");
        document.removeEventListener("keydown", escHandler);
      }
    };
    document.addEventListener("keydown", escHandler);
  }

  function removeOverlay() {
    if (overlay) {
      overlay.remove();
      overlay = null;
    }
  }

  // Main click interceptor
  document.addEventListener(
    "click",
    (e) => {
      if (!enabled) return;

      // Don't intercept our own overlay clicks
      if (isOverlayClick(e.target)) return;

      // Don't intercept if this is our re-dispatched click
      if (pendingTarget && (e.target === pendingTarget || pendingTarget.contains(e.target))) {
        pendingTarget = null;
        return;
      }

      // Find the clickable element
      const clickable = findClickableAncestor(e.target);
      if (!clickable) return;

      // Intercept!
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();

      showConfirmation(clickable, e);
    },
    true // capture phase - intercept before anything else
  );

  console.log("[Schwab Trade Confirm] Extension loaded and active");
})();
