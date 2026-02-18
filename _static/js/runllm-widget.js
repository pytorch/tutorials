/**
 * RunLLM Widget Integration
 *
 * This script loads the RunLLM widget with configurable options.
 * Configuration is passed from Sphinx conf.py via html_theme_options.
 *
 * Usage in conf.py:
 *   html_theme_options = {
 *       "runllm_assistant_id": "834",
 *       "runllm_name": "PyTorch",
 *       "runllm_position": "BOTTOM_RIGHT",
 *       "runllm_keyboard_shortcut": "Mod+j",
 *   }
 */

document.addEventListener("DOMContentLoaded", function () {
  // Get configuration from window.runllmConfig (set by theme template)
  var config = window.runllmConfig || {};

  // Only load widget if assistant_id is configured
  if (!config.assistant_id) {
    console.log("RunLLM widget: No assistant_id configured, skipping widget load");
    return;
  }

  var script = document.createElement("script");
  script.type = "module";
  script.id = "runllm-widget-script";
  script.src = "https://widget.runllm.com";

  script.setAttribute("version", "stable");
  script.setAttribute("crossorigin", "true");
  script.setAttribute("runllm-keyboard-shortcut", config.keyboard_shortcut || "Mod+j");
  script.setAttribute("runllm-name", config.name || "Assistant");
  script.setAttribute("runllm-position", config.position || "BOTTOM_RIGHT");
  script.setAttribute("runllm-assistant-id", config.assistant_id);

  script.async = true;
  document.head.appendChild(script);
});
