"""
One-off migration: replace the per-page copies of the sidebar, footer, theme
handling and mobile menu with calls into shared/toolkit.js.

Run once from the repo root:  python3 build/extract_shell.py
Kept in the repo so the transformation is auditable rather than a mystery diff.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PAGES = {
    "index.html": "overview",
    "legal.html": None,
    "settings.html": None,
    "user-guide.html": None,
    "tools/word-embeddings.html": "word-embeddings",
    "tools/explainability-lab.html": "explainability-lab",
    "tools/bias-auditor.html": "bias-auditor",
    "tools/adversarial-sandbox.html": "adversarial-sandbox",
    "tools/filter-bubble.html": "filter-bubble",
    "tools/privacy-lab.html": "privacy-lab",
    "tools/value-alignment.html": "value-alignment",
    "tools/proxy-detector.html": "proxy-detector",
    "tools/llm-sandbox.html": "llm-sandbox",
}

# Resolves and paints the theme before first paint. This must stay inline in
# <head>: as an external file it would load after the initial render and
# light-mode users would see the page flash dark on every navigation.
THEME_SNIPPET = """    <script>
        // Paint the saved theme before first render to avoid a flash of the
        // wrong theme. Toolkit.mountShell() wires the toggle control later.
        (function () {
            var saved = null;
            try { saved = localStorage.getItem('ethicsToolkitTheme'); } catch (e) { }
            function apply(dark) {
                var root = document.documentElement;
                root.classList.toggle('dark', dark);
                root.classList.toggle('light', !dark);
            }
            apply(saved !== 'light');
            window.__setToolkitTheme = function (dark) {
                apply(dark);
                try { localStorage.setItem('ethicsToolkitTheme', dark ? 'dark' : 'light'); } catch (e) { }
            };
        })();
    </script>
"""


def rel(path: str) -> str:
    return "../" if path.startswith("tools/") else ""


def strip_sidebar(text: str) -> tuple[str, bool]:
    """Remove the mobile overlay + <aside> block; the shell injects them."""
    start = text.find('<div id="mobileOverlay"')
    if start == -1:
        return text, False
    # Include the preceding comment line when present.
    line_start = text.rfind("\n", 0, start) + 1
    comment = text.rfind("<!-- Mobile Overlay -->", 0, start)
    if comment != -1 and start - comment < 120:
        line_start = text.rfind("\n", 0, comment) + 1
    end = text.find("</aside>", start)
    if end == -1:
        return text, False
    end = text.find("\n", end) + 1
    return text[:line_start] + text[end:], True


def strip_footer(text: str) -> tuple[str, bool]:
    """Remove the disclaimer footer that is a direct child of <main>.

    Matched on its full markup rather than on the <footer> tag alone, because
    bias-auditor.html also emits a <footer> inside the exported-report template
    string and that one must survive.
    """
    pattern = re.compile(
        r"[ \t]*(?:<!-- Disclaimer Footer -->\s*\n)?[ \t]*<footer class=\"mt-(?:auto|8) py-6[^\"]*\">.*?</footer>\s*\n",
        re.DOTALL,
    )
    new, count = pattern.subn("", text, count=1)
    return new, count == 1


def strip_inline_scripts(text: str) -> tuple[str, list[str]]:
    """Remove the duplicated theme and mobile-menu blocks from page scripts."""
    removed = []

    # Whole trailing <script> blocks that contain nothing but the shell wiring.
    def only_shell(body: str) -> bool:
        meaningful = [
            ln for ln in body.splitlines()
            if ln.strip() and not ln.strip().startswith("//")
        ]
        if not meaningful:
            return False
        joined = "\n".join(meaningful)
        return ("toggleSidebar" in joined or "setTheme" in joined) and not re.search(
            r"function (?!toggleSidebar|setTheme)\w+", joined
        )

    out = []
    pos = 0
    for m in re.finditer(r"[ \t]*<script>\n(.*?)</script>\s*\n", text, re.DOTALL):
        if only_shell(m.group(1)):
            out.append(text[pos:m.start()])
            pos = m.end()
            removed.append("shell-only script block")
    out.append(text[pos:])
    text = "".join(out)

    # Theme wiring left inside a larger script block.
    patterns = [
        (r"[ \t]*// =+ THEME =+\n(?:[ \t]*(?://.*)?\n)*?"
         r"[ \t]*const themeToggle[^\n]*\n"
         r"(?:[ \t]*(?:function setTheme|const savedTheme|if \(savedTheme|themeToggle\.addEventListener)[^\n]*\n)+",
         "theme block"),
        (r"[ \t]*// Theme\n"
         r"[ \t]*const themeToggle[^\n]*\n"
         r"(?:[ \t]*(?:function setTheme|const savedTheme|if \(localStorage|if \(savedTheme|themeToggle\.addEventListener)[^\n]*\n)+",
         "theme block"),
        (r"[ \t]*// =+ MOBILE MENU =+\n"
         r"(?:.*?\n)*?[ \t]*mobileOverlay\??\.addEventListener\('click', toggleSidebar\);\n",
         "mobile menu block"),
    ]
    for pat, label in patterns:
        text, n = re.subn(pat, "", text, count=1)
        if n:
            removed.append(label)
    return text, removed


def inject_head(text: str, path: str) -> str:
    base = rel(path)
    if "__setToolkitTheme" not in text:
        anchor = text.index("</head>")
        text = text[:anchor] + THEME_SNIPPET + text[anchor:]
    link = f'    <link rel="stylesheet" href="{base}shared/toolkit.css" />\n'
    if "shared/toolkit.css" not in text:
        anchor = text.index("</head>")
        text = text[:anchor] + link + text[anchor:]
    if "rel=\"icon\"" not in text:
        icon = ('    <link rel="icon"'
                ' href="data:image/svg+xml,'
                "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E"
                "%3Ctext y='.9em' font-size='90'%3E%F0%9F%A7%AD%3C/text%3E%3C/svg%3E\" />\n")
        anchor = text.index("</head>")
        text = text[:anchor] + icon + text[anchor:]
    return text


def inject_scripts(text: str, path: str, active) -> str:
    base = rel(path)
    active_js = "null" if active is None else f"'{active}'"
    block = (
        f'    <script src="{base}shared/session.js"></script>\n'
        f'    <script src="{base}shared/toolkit.js"></script>\n'
        f'    <script>\n'
        f'        Toolkit.mountShell({{ active: {active_js} }});\n'
        f'        ToolkitSession.maybeOfferConsent();\n'
        f'    </script>\n'
    )
    anchor = text.rindex("</body>")
    return text[:anchor] + block + text[anchor:]


def main() -> int:
    failures = []
    for page, active in PAGES.items():
        path = ROOT / page
        text = original = path.read_text()

        text, ok_side = strip_sidebar(text)
        text, ok_foot = strip_footer(text)
        text, removed = strip_inline_scripts(text)
        text = inject_head(text, page)
        text = inject_scripts(text, page, active)

        if not ok_side:
            failures.append(f"{page}: sidebar not found")
        if not ok_foot:
            failures.append(f"{page}: footer not found")

        path.write_text(text)
        delta = len(original.splitlines()) - len(text.splitlines())
        print(f"  {page:38s} -{delta:4d} lines   removed: {', '.join(removed) or 'none'}")

    if failures:
        print("\nPROBLEMS:")
        for f in failures:
            print("  " + f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
