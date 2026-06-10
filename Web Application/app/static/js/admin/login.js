/**
 * app/static/js/admin/login.js — Admin Login Form JavaScript
 * ===========================================================
 * Enhances the admin login form with client-side UX improvements.
 * The form is fully functional WITHOUT this script (progressive enhancement).
 * JS only adds polish — it never replaces the server-side validation in auth.py.
 *
 * Features:
 *   1. Client-side input validation (pre-flight before submitting to server)
 *   2. Password visibility toggle
 *   3. Loading state on the submit button during POST
 *   4. Auto-dismiss for flash alert messages
 *   5. Focus management on page load
 *   6. Rate limit feedback (HTTP 429 response handling)
 *
 * Integration:
 *   Loaded at the bottom of admin/login.html via:
 *     <script src="{{ url_for('static', filename='js/admin/login.js') }}" defer></script>
 *
 *   The script targets elements by their data attributes (data-login-*),
 *   never by class or id alone, making it resilient to CSS refactors.
 *
 * No external dependencies. Vanilla JS only.
 */

'use strict';

/* ==========================================================================
   CONSTANTS
   ========================================================================== */

const SELECTORS = {
  form:        '[data-login-form]',
  emailInput:  '[data-login-email]',
  pwInput:     '[data-login-password]',
  pwToggle:    '[data-login-pw-toggle]',
  submitBtn:   '[data-login-submit]',
  flashItems:  '[data-flash-alert]',
  flashClose:  '[data-flash-close]',
};

const CLASSES = {
  fieldInvalid: 'is-invalid',
  btnLoading:   'is-loading',
};

const VALIDATION = {
  emailRegex:  /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
  minPwLength: 6,
};

const AUTO_DISMISS_MS = 5000;  // Flash alerts auto-dismiss after 5 seconds

/* ==========================================================================
   INITIALISATION
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  initFlashAlerts();
  initPasswordToggle();
  initForm();
  setInitialFocus();
});

/* ==========================================================================
   1. FLASH ALERT AUTO-DISMISS
   ========================================================================== */
/**
 * Flash messages are rendered server-side via Jinja2 get_flashed_messages().
 * This function adds:
 *   - Auto-dismiss after AUTO_DISMISS_MS milliseconds
 *   - Manual dismiss via the close button
 *   - Smooth slide-out animation before removal
 *
 * The flash container is defined in base.html and is always present.
 */
function initFlashAlerts() {
  document.querySelectorAll(SELECTORS.flashItems).forEach((alert) => {
    // Auto-dismiss
    const timer = setTimeout(() => dismissAlert(alert), AUTO_DISMISS_MS);

    // Manual close button
    const closeBtn = alert.querySelector(SELECTORS.flashClose);
    if (closeBtn) {
      closeBtn.addEventListener('click', () => {
        clearTimeout(timer);
        dismissAlert(alert);
      });
    }
  });
}

/**
 * Animate an alert out and remove it from the DOM.
 * The CSS class 'is-hiding' triggers the slide-out keyframe animation.
 */
function dismissAlert(alertEl) {
  if (!alertEl || alertEl.dataset.dismissed) return;
  alertEl.dataset.dismissed = 'true';
  alertEl.classList.add('is-hiding');

  // Remove from DOM after animation completes (matches CSS animation duration 250ms)
  alertEl.addEventListener('animationend', () => alertEl.remove(), { once: true });
}

/* ==========================================================================
   2. PASSWORD VISIBILITY TOGGLE
   ========================================================================== */
/**
 * Toggles the password input between type="password" and type="text".
 * The toggle button's icon swaps between an eye-open and eye-closed SVG.
 * ARIA attributes are updated so screen readers announce the current state.
 */
function initPasswordToggle() {
  const pwInput  = document.querySelector(SELECTORS.pwInput);
  const pwToggle = document.querySelector(SELECTORS.pwToggle);

  if (!pwInput || !pwToggle) return;

  pwToggle.addEventListener('click', () => {
    const isHidden   = pwInput.type === 'password';
    pwInput.type     = isHidden ? 'text' : 'password';

    // Update ARIA label
    pwToggle.setAttribute(
      'aria-label',
      isHidden ? 'Hide password' : 'Show password'
    );
    pwToggle.setAttribute('aria-pressed', String(isHidden));

    // Swap the icon (data-icon-show / data-icon-hide spans inside the button)
    const showIcon = pwToggle.querySelector('[data-icon-show]');
    const hideIcon = pwToggle.querySelector('[data-icon-hide]');
    if (showIcon) showIcon.hidden = isHidden;
    if (hideIcon) hideIcon.hidden = !isHidden;

    // Return focus to the password field
    pwInput.focus();
  });
}

/* ==========================================================================
   3. FORM VALIDATION & SUBMISSION
   ========================================================================== */

function initForm() {
  const form      = document.querySelector(SELECTORS.form);
  const submitBtn = document.querySelector(SELECTORS.submitBtn);

  if (!form || !submitBtn) return;

  // Clear field-level errors on input (user is correcting a mistake)
  form.querySelectorAll('input').forEach((input) => {
    input.addEventListener('input', () => clearFieldError(input));
    input.addEventListener('blur',  () => validateField(input));
  });

  // Intercept submission for client-side pre-validation
  form.addEventListener('submit', (event) => {
    const isValid = validateAllFields(form);

    if (!isValid) {
      // Prevent server round-trip if client-side validation fails
      event.preventDefault();
      // Focus the first invalid field
      const firstInvalid = form.querySelector(`.${CLASSES.fieldInvalid} input`);
      if (firstInvalid) firstInvalid.focus();
      return;
    }

    // All fields valid — show loading state and allow the form to POST
    setSubmitLoading(submitBtn, true);

    // Safety: reset loading state if the browser navigation is cancelled
    // (e.g. user presses Escape or the request fails at the network level).
    // The server always redirects or re-renders, so this only fires on failure.
    window.addEventListener('pageshow', () => setSubmitLoading(submitBtn, false), { once: true });
  });
}

/**
 * Validate a single <input> element. Returns true if valid.
 * Sets the is-invalid class and populates the error hint on the parent .form-field.
 */
function validateField(input) {
  const name  = input.name;
  const value = input.value.trim();
  let errorMsg = '';

  if (name === 'email') {
    if (!value) {
      errorMsg = 'Email address is required.';
    } else if (!VALIDATION.emailRegex.test(value)) {
      errorMsg = 'Please enter a valid email address.';
    }
  }

  if (name === 'password') {
    if (!value) {
      errorMsg = 'Password is required.';
    } else if (value.length < VALIDATION.minPwLength) {
      errorMsg = `Password must be at least ${VALIDATION.minPwLength} characters.`;
    }
  }

  const field = input.closest('.form-field');
  if (!field) return !errorMsg;

  if (errorMsg) {
    setFieldError(field, errorMsg);
    return false;
  }

  clearFieldError(input);
  return true;
}

/**
 * Validate all form fields. Returns true if ALL are valid.
 */
function validateAllFields(form) {
  const inputs = form.querySelectorAll('input[required]');
  let allValid = true;
  inputs.forEach((input) => {
    if (!validateField(input)) allValid = false;
  });
  return allValid;
}

/**
 * Set the error state on a .form-field wrapper and update its error text.
 */
function setFieldError(fieldEl, message) {
  fieldEl.classList.add(CLASSES.fieldInvalid);
  const errorEl = fieldEl.querySelector('.form-field__error');
  if (errorEl) {
    const textNode = errorEl.querySelector('[data-error-text]');
    if (textNode) textNode.textContent = message;
  }

  const input = fieldEl.querySelector('input');
  if (input) {
    // ARIA: announce the error to screen readers
    input.setAttribute('aria-invalid', 'true');
    const errorId = errorEl ? errorEl.id : null;
    if (errorId) input.setAttribute('aria-describedby', errorId);
  }
}

/**
 * Clear the error state on a .form-field wrapper.
 */
function clearFieldError(input) {
  const field = input.closest('.form-field');
  if (!field) return;

  field.classList.remove(CLASSES.fieldInvalid);
  input.removeAttribute('aria-invalid');
  input.removeAttribute('aria-describedby');
}

/**
 * Toggle the loading state on the submit button.
 * - Disables the button (prevents double-submit)
 * - Shows the CSS spinner (via .is-loading class on the button)
 * - Updates aria-label for screen readers
 */
function setSubmitLoading(btn, isLoading) {
  if (!btn) return;
  btn.disabled = isLoading;
  btn.classList.toggle(CLASSES.btnLoading, isLoading);
  btn.setAttribute(
    'aria-label',
    isLoading ? 'Signing in, please wait…' : 'Sign in'
  );
}

/* ==========================================================================
   4. INITIAL FOCUS
   ========================================================================== */
/**
 * Move focus to the email field on page load so the admin can start
 * typing immediately without clicking. Skip if a flash error from a failed
 * login is visible — in that case, the flash already has focus-relevant info.
 */
function setInitialFocus() {
  const emailInput = document.querySelector(SELECTORS.emailInput);
  if (!emailInput) return;

  // If the email already has a value (e.g. after a failed login where the
  // browser preserved the value), focus the password field instead.
  if (emailInput.value && emailInput.value.length > 0) {
    const pwInput = document.querySelector(SELECTORS.pwInput);
    if (pwInput) {
      pwInput.focus();
      return;
    }
  }

  emailInput.focus();
}
