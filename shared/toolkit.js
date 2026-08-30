/**
 * Shared chrome for every page in the AI Ethics Toolkit.
 *
 * Before this existed the sidebar, footer, theme handling and mobile menu were
 * copy-pasted into ten files and had already drifted apart: setTheme differed
 * between pages, and the light-mode scrollbar rule existed only on index.html,
 * so every tool page rendered a dark scrollbar against a white background.
 *
 * Pages now supply only their own <main> content and call:
 *
 *     Toolkit.mountShell({ active: 'bias-auditor' });
 *
 * Theme resolution deliberately does NOT live here. It runs from a small
 * inline script in <head> so the correct theme is painted before first render;
 * loading it as an external file would reintroduce the flash of dark content
 * that light-mode users used to see on every page load.
 */
(function (global) {
    'use strict';

    // ---------------------------------------------------------------- tools

    const TOOLS = [
        {
            id: 'word-embeddings',
            name: 'Word Embeddings Workbench',
            short: 'Word Embeddings',
            icon: 'scatter_plot',
            color: 'blue',
            category: 'word-spaces',
            file: 'word-embeddings.html',
            lab: 'Lab 1',
            description: 'Explore vector arithmetic, measure gender bias with WEAT, and watch debiasing fail to remove it.'
        },
        {
            id: 'explainability-lab',
            name: 'Model Explainability Lab',
            short: 'Explainability Lab',
            icon: 'visibility',
            color: 'purple',
            category: 'explainability',
            file: 'explainability-lab.html',
            lab: 'Lab 7',
            description: 'Compare a sampled LIME surrogate against exact Shapley values on text, lending, health and risk-scoring models.'
        },
        {
            id: 'bias-auditor',
            name: 'Dataset Bias Auditor',
            short: 'Bias Auditor',
            icon: 'saved_search',
            color: 'orange',
            category: 'fairness',
            file: 'bias-auditor.html',
            lab: 'Lab 3',
            description: 'Compute selection-rate and error-rate fairness metrics from a real confusion matrix, and see why they conflict.'
        },
        {
            id: 'adversarial-sandbox',
            name: 'Adversarial Robustness Sandbox',
            short: 'Adversarial Sandbox',
            icon: 'security',
            color: 'red',
            category: 'safety',
            file: 'adversarial-sandbox.html',
            lab: 'Lab 9',
            description: 'Run gradient-based FGSM attacks against MobileNet v2 in the browser and find the perturbation budget that breaks it.'
        },
        {
            id: 'filter-bubble',
            name: 'Filter Bubble Simulator',
            short: 'Filter Bubble Sim',
            icon: 'bubble_chart',
            color: 'teal',
            category: 'fairness',
            file: 'filter-bubble.html',
            lab: 'Lab 4',
            description: 'Watch an engagement-optimised feed narrow over time, with a chronological control to compare against.'
        },
        {
            id: 'privacy-lab',
            name: 'Privacy & Anonymization Lab',
            short: 'Privacy Lab',
            icon: 'vpn_key',
            color: 'indigo',
            category: 'privacy',
            file: 'privacy-lab.html',
            lab: 'Labs 5–6',
            description: 'Search the k-anonymity generalization lattice and spend a differential privacy budget until it runs out.'
        },
        {
            id: 'value-alignment',
            name: 'Value Alignment Tool',
            short: 'Value Alignment',
            icon: 'balance',
            color: 'pink',
            category: 'safety',
            file: 'value-alignment.html',
            lab: 'Lab 10',
            description: 'Work through 12 real AI dilemmas and see your choices mapped across five ethical frameworks.'
        },
        {
            id: 'proxy-detector',
            name: 'Proxy Variable Detector',
            short: 'Proxy Detector',
            icon: 'find_replace',
            color: 'cyan',
            category: 'fairness',
            file: 'proxy-detector.html',
            lab: 'Lab 2',
            description: 'Measure how strongly ordinary features encode protected attributes, using the right statistic for each pair.'
        },
        {
            id: 'llm-sandbox',
            name: 'LLM Ethical Sandbox',
            short: 'LLM Ethical Sandbox',
            icon: 'chat',
            color: 'emerald',
            category: 'safety',
            file: 'llm-sandbox.html',
            lab: 'Lab 11',
            description: 'Red-team a small language model running entirely in your browser via WebGPU.'
        }
    ];

    const CATEGORIES = {
        'word-spaces': 'Word Spaces',
        'explainability': 'Explainability',
        'fairness': 'Fairness Lab',
        'privacy': 'Privacy',
        'safety': 'Safety'
    };

    const VERSION = '3.0.0';

    // Pages live either at the repo root or one level down in /tools.
    const inToolsDir = /\/tools\//.test(global.location.pathname);
    const base = inToolsDir ? '../' : '';
    const toolHref = tool => (inToolsDir ? '' : 'tools/') + tool.file;

    function esc(value) {
        return String(value ?? '').replace(/[&<>"']/g, c => (
            { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
        ));
    }

    // ---------------------------------------------------------------- shell

    function sidebarMarkup(active) {
        const item = (href, icon, label, isActive, extra = '') => `
            <a href="${href}"
               ${isActive ? 'aria-current="page"' : ''}
               class="flex items-center px-3 py-2.5 rounded-lg text-sm transition-colors group ${isActive
                    ? 'bg-primary/10 text-primary font-medium'
                    : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-surface-dark-lighter hover:text-slate-900 dark:hover:text-white'}">
                <span class="material-icons-outlined mr-3 text-xl ${isActive ? '' : 'group-hover:text-primary transition-colors'}"
                      aria-hidden="true">${icon}</span>
                <span>${esc(label)}</span>${extra}
            </a>`;

        return `
        <div class="fixed inset-0 bg-black/50 z-40 hidden transition-opacity opacity-0" id="mobileOverlay" hidden></div>
        <aside id="sidebar" aria-label="Toolkit navigation"
            class="fixed inset-y-0 left-0 z-50 w-64 bg-white dark:bg-surface-dark border-r border-slate-200 dark:border-slate-800 flex flex-col transition-transform duration-300 transform -translate-x-full md:translate-x-0">
            <div class="h-16 flex items-center justify-between px-6 border-b border-slate-200 dark:border-slate-800">
                <a href="${base}index.html" class="flex items-center hover:opacity-80 transition-opacity">
                    <span class="material-icons-outlined text-primary text-3xl mr-2" aria-hidden="true">psychology</span>
                    <span class="text-xl font-bold tracking-tight dark:text-white">AI Ethics Toolkit</span>
                </a>
                <button id="closeSidebar" type="button" aria-label="Close navigation"
                    class="md:hidden text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 rounded focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary">
                    <span class="material-icons-outlined" aria-hidden="true">close</span>
                </button>
            </div>
            <nav class="flex-1 overflow-y-auto py-6 px-3 space-y-1" aria-label="Tools">
                ${item(base + 'index.html', 'grid_view', 'Toolkit Overview', active === 'overview')}
                <div class="pt-4 pb-2 px-3">
                    <h2 class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Tools</h2>
                </div>
                ${TOOLS.map(t => item(toolHref(t), t.icon, t.short, active === t.id)).join('')}
            </nav>
            <div class="border-t border-slate-200 dark:border-slate-800 p-4">
                <p class="text-[10px] text-slate-400 mb-2">© 2026
                    <a href="https://hamedyaghoobian.com" class="text-primary hover:underline"
                       target="_blank" rel="noopener">Hamed Yaghoobian</a></p>
                <div class="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
                    <span>Version ${VERSION}</span>
                    <a class="hover:text-primary transition-colors" href="${base}legal.html">Legal</a>
                </div>
            </div>
        </aside>`;
    }

    function footerMarkup() {
        return `
        <footer class="mt-auto pt-6 border-t border-slate-200 dark:border-slate-800">
            <div class="text-center text-xs text-slate-400 dark:text-slate-500 px-4 leading-relaxed">
                <p>AI Ethics Toolkit &copy; 2026 Hamed Yaghoobian.<br>
                    Developed for the Department of Mathematics, Computer Science, and Statistics at Muhlenberg College.<br>
                    Feedback: <a href="mailto:hamedyaghoobian@muhlenberg.edu"
                        class="hover:text-primary transition-colors">hamedyaghoobian@muhlenberg.edu</a></p>
            </div>
        </footer>`;
    }

    // ------------------------------------------------------------- behaviour

    function wireMobileMenu() {
        const sidebar = document.getElementById('sidebar');
        const overlay = document.getElementById('mobileOverlay');
        const openBtn = document.getElementById('openSidebar');
        const closeBtn = document.getElementById('closeSidebar');
        if (!sidebar || !overlay) return;

        let open = false;
        function setOpen(next) {
            open = next;
            sidebar.classList.toggle('-translate-x-full', !open);
            if (open) {
                overlay.hidden = false;
                overlay.classList.remove('hidden');
                requestAnimationFrame(() => overlay.classList.remove('opacity-0'));
            } else {
                overlay.classList.add('opacity-0');
                setTimeout(() => { overlay.classList.add('hidden'); overlay.hidden = true; }, 300);
            }
            if (openBtn) openBtn.setAttribute('aria-expanded', String(open));
            if (open) closeBtn?.focus(); else openBtn?.focus();
        }

        openBtn?.addEventListener('click', () => setOpen(true));
        closeBtn?.addEventListener('click', () => setOpen(false));
        overlay.addEventListener('click', () => setOpen(false));
        document.addEventListener('keydown', e => { if (e.key === 'Escape' && open) setOpen(false); });

        if (openBtn) {
            openBtn.setAttribute('aria-expanded', 'false');
            openBtn.setAttribute('aria-controls', 'sidebar');
        }
    }

    // The head script owns initial paint; this only wires the toggle control.
    function wireThemeToggle() {
        const toggle = document.getElementById('themeToggle');
        if (!toggle) return;
        toggle.checked = document.documentElement.classList.contains('dark');
        toggle.addEventListener('change', () => {
            global.__setToolkitTheme(toggle.checked);
            document.dispatchEvent(new CustomEvent('toolkit:themechange', {
                detail: { dark: toggle.checked }
            }));
        });
    }

    function mountShell(options = {}) {
        const { active = null } = options;

        document.body.insertAdjacentHTML('afterbegin', sidebarMarkup(active));

        const main = document.querySelector('main');
        if (main && !main.querySelector('footer')) {
            main.insertAdjacentHTML('beforeend', footerMarkup());
        }

        wireMobileMenu();
        wireThemeToggle();

        if (global.ToolkitSession && active && active !== 'overview') {
            global.ToolkitSession.startTool(active);
        }
    }

    global.Toolkit = { TOOLS, CATEGORIES, VERSION, mountShell, esc, base, toolHref };
})(window);
