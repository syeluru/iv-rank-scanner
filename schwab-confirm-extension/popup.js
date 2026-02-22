const toggle = document.getElementById("enableToggle");
const statusText = document.getElementById("statusText");

// Load current state
chrome.storage.local.get("schwabConfirmEnabled", (data) => {
  const isEnabled = data.schwabConfirmEnabled !== false;
  toggle.checked = isEnabled;
  updateStatus(isEnabled);
});

// Handle toggle
toggle.addEventListener("change", () => {
  const isEnabled = toggle.checked;
  chrome.storage.local.set({ schwabConfirmEnabled: isEnabled });
  updateStatus(isEnabled);
});

function updateStatus(enabled) {
  statusText.textContent = enabled ? "Protection Active" : "Protection Disabled";
  statusText.className = "status" + (enabled ? " active" : "");
}
