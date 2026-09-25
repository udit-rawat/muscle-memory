"""In-page scripts used by the web surface. Kept separate so the Python stays readable."""

# Candidate elements for an observation: things you can act on, plus table cells you can read.
INTERACTIVE_SELECTOR = (
    "a[href], button, input:not([type=hidden]), select, textarea, "
    "[role=button], [role=link], [role=checkbox], [role=tab], [onclick]"
)
READABLE_SELECTOR = "td"

# Describe one element: role, accessible name (approximate), attributes useful for locators.
# The name is only a hint; every locator built from it is verified against the live element.
DESCRIBE = r"""
(el) => {
  const txt = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const tag = el.tagName.toLowerCase();
  const type = (el.getAttribute('type') || '').toLowerCase();
  const BUTTON_TYPES = ['submit', 'button', 'reset', 'image'];
  let role = el.getAttribute('role') || '';
  if (!role) {
    if (tag === 'a' && el.hasAttribute('href')) role = 'link';
    else if (tag === 'button' || (tag === 'input' && BUTTON_TYPES.includes(type))) role = 'button';
    else if (tag === 'input' && type === 'checkbox') role = 'checkbox';
    else if (tag === 'input' && type === 'radio') role = 'radio';
    else if (tag === 'select') role = (el.multiple || el.size > 1) ? 'listbox' : 'combobox';
    else if (tag === 'textarea' || tag === 'input') role = 'textbox';
    else if (tag === 'td') role = 'cell';
    else role = 'generic';
  }
  let name = txt(el.getAttribute('aria-label'));
  if (!name && el.labels && el.labels.length) name = txt(el.labels[0].innerText);
  if (!name && role === 'button' && tag === 'input') name = txt(el.value);
  if (!name && ['link', 'button', 'cell', 'generic'].includes(role)) name = txt(el.innerText);
  if (!name) name = txt(el.getAttribute('title'));
  if (!name) name = txt(el.getAttribute('placeholder'));

  const r = el.getBoundingClientRect();
  const style = getComputedStyle(el);
  const visible = r.width > 0 && r.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';

  const parts = [];
  for (let n = el; n && n.nodeType === 1 && n !== document.documentElement; n = n.parentElement) {
    let i = 1;
    for (let s = n.previousElementSibling; s; s = s.previousElementSibling) if (s.tagName === n.tagName) i++;
    parts.unshift(n.tagName.toLowerCase() + ':nth-of-type(' + i + ')');
  }

  let value = null;
  const hideValue = BUTTON_TYPES.concat(['password', 'checkbox', 'radio']).includes(type);
  if (tag === 'input' && !hideValue) value = el.value;
  if (tag === 'textarea') value = el.value;
  if (tag === 'select') value = el.selectedOptions[0] ? txt(el.selectedOptions[0].text) : '';
  const options = tag === 'select' ? Array.from(el.options).map((o) => txt(o.text)) : null;
  const nested = tag === 'td' && !!el.querySelector('a, input, select, button, textarea, table');

  return {
    tag, type, role, name, visible, value, options, nested,
    title: txt(el.getAttribute('title')),
    text: txt(el.innerText).slice(0, 80),
    name_attr: el.getAttribute('name') || '',
    css: parts.join(' > '),
  };
}
"""

# For a table cell being read: its column index, the table's header texts, and the other cells
# in its row (candidate row keys). Lets us target "Balance of the Share Savings row" instead of
# the value itself.
CELL_CONTEXT = r"""
(el) => {
  const txt = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const cells = Array.from(el.parentElement.children);
  const idx = cells.indexOf(el);
  const table = el.closest('table');
  const hdr = Array.from(table.rows).find(
    (r) => r.children.length && Array.from(r.children).every((c) => c.tagName === 'TH'));
  const headers = hdr ? Array.from(hdr.children).map((c) => txt(c.innerText)) : null;
  const keys = cells
    .filter((c, i) => i !== idx && c.tagName === 'TD')
    .map((c) => txt(c.innerText))
    .filter((t) => t);
  return { idx, headers, keys };
}
"""

BODY_TEXT = "() => document.body ? document.body.innerText : ''"

# Returns the text of the topmost large fixed/absolute layer in this frame, or null if none.
# "Large" = covers at least 25% of the frame's viewport; the text is taken from the layer itself or,
# for a bare backdrop, from any other fixed layer in the frame (the dialog box sitting on it).
BLOCKING_OVERLAY = r"""
() => {
  if (!document.body) return null;
  const vw = window.innerWidth, vh = window.innerHeight;
  const layers = Array.from(document.body.querySelectorAll('*')).filter((el) => {
    const s = getComputedStyle(el);
    if (s.position !== 'fixed' || s.display === 'none' || s.visibility === 'hidden') return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  });
  const big = layers.find((el) => {
    const r = el.getBoundingClientRect();
    const w = Math.min(r.right, vw) - Math.max(r.left, 0), h = Math.min(r.bottom, vh) - Math.max(r.top, 0);
    return w * h >= 0.25 * vw * vh;
  });
  if (!big) return null;
  const text = layers.map((el) => (el.innerText || '').trim()).filter((t) => t).join(' ');
  return text || '(untitled overlay)';
}
"""
