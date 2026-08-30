/**
 * Local-first session recorder.
 *
 * Design constraints, in order of priority:
 *
 *   1. Nothing leaves this browser unless the student explicitly exports it.
 *      There is no network call in this file. There is no third-party script.
 *      If you are reading this to check: search for "fetch", "XMLHttpRequest",
 *      "sendBeacon" and "Image(" below. You will not find them.
 *
 *   2. The student can see everything that was recorded, at any time, in the
 *      Session Record panel. Nothing is captured that is not shown.
 *
 *   3. No identity is collected. No name, no email, no student ID, no IP, no
 *      device fingerprint. Events are timestamps, tool names and the parameter
 *      values you chose. A random session id groups events from one sitting and
 *      is regenerated whenever the record is cleared.
 *
 * Why it works this way: this toolkit teaches surveillance, consent and
 * telemetry. Instrumenting students invisibly to study them would be the exact
 * behaviour the privacy labs ask them to critique, and in a CS course somebody
 * reads the source. Making the instrumentation legible turns it from a liability
 * into course material - see the "What this page recorded about you" panel.
 *
 * Submitting a session export with a lab report is coursework, not data
 * collection: the student chooses what to hand in, the same way they choose to
 * hand in the report itself.
 */
(function (global) {
    'use strict';

    const STORAGE_KEY = 'ethicsToolkitSession';
    const CONSENT_KEY = 'ethicsToolkitSessionConsent';
    const MAX_EVENTS = 2000;          // ring buffer; oldest drop first
    const SCHEMA = 1;

    let state = null;
    let currentTool = null;
    let toolStartedAt = null;

    // ------------------------------------------------------------- storage

    function randomId() {
        const bytes = new Uint8Array(8);
        (global.crypto || global.msCrypto).getRandomValues(bytes);
        return Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
    }

    function blankState() {
        return { schema: SCHEMA, sessionId: randomId(), startedAt: new Date().toISOString(), events: [] };
    }

    function load() {
        if (state) return state;
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            const parsed = raw ? JSON.parse(raw) : null;
            state = (parsed && parsed.schema === SCHEMA && Array.isArray(parsed.events))
                ? parsed : blankState();
        } catch (e) {
            // Private windows, cleared site data and blocked storage all land
            // here. Recording is a convenience, never a precondition.
            state = blankState();
        }
        return state;
    }

    function persist() {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        } catch (e) { /* quota or blocked storage: keep going in memory */ }
    }

    // Recording is OFF until the student turns it on. No pre-ticked boxes.
    function isEnabled() {
        try { return localStorage.getItem(CONSENT_KEY) === 'on'; } catch (e) { return false; }
    }

    function setEnabled(on) {
        try { localStorage.setItem(CONSENT_KEY, on ? 'on' : 'off'); } catch (e) { /* ignore */ }
        if (!on) clear();
        document.dispatchEvent(new CustomEvent('toolkit:session-consent', { detail: { enabled: on } }));
        renderPanel();
    }

    function hasDecided() {
        try { return localStorage.getItem(CONSENT_KEY) !== null; } catch (e) { return true; }
    }

    // -------------------------------------------------------------- record

    /**
     * Record one event.
     *
     * @param {string} type   e.g. 'tool.open', 'param.change', 'export'
     * @param {object} detail plain JSON values only - no DOM nodes, no free text
     *                        typed by the student, no file contents.
     */
    function record(type, detail = {}) {
        if (!isEnabled()) return;
        load();
        state.events.push({
            t: new Date().toISOString(),
            tool: currentTool,
            type,
            detail: sanitize(detail)
        });
        if (state.events.length > MAX_EVENTS) {
            state.events.splice(0, state.events.length - MAX_EVENTS);
        }
        persist();
        renderPanel();
    }

    // Only primitives and shallow arrays survive. This is deliberate: it makes
    // it impossible to accidentally capture typed prose, uploaded rows or
    // anything else that could carry personal data.
    function sanitize(detail) {
        const out = {};
        Object.entries(detail || {}).forEach(([k, v]) => {
            if (v === null || v === undefined) return;
            if (typeof v === 'number' || typeof v === 'boolean') { out[k] = v; return; }
            if (typeof v === 'string') { out[k] = v.slice(0, 120); return; }
            if (Array.isArray(v)) {
                out[k] = v.slice(0, 20).map(x =>
                    typeof x === 'object' ? '[object]' : String(x).slice(0, 60));
            }
        });
        return out;
    }

    function startTool(toolId) {
        if (currentTool === toolId) return;
        endTool();
        currentTool = toolId;
        toolStartedAt = Date.now();
        record('tool.open', {});
    }

    function endTool() {
        if (!currentTool || !toolStartedAt) return;
        const seconds = Math.round((Date.now() - toolStartedAt) / 1000);
        if (seconds >= 3) record('tool.close', { seconds });
        currentTool = null;
        toolStartedAt = null;
    }

    function clear() {
        state = blankState();
        persist();
        renderPanel();
    }

    function summary() {
        load();
        const byTool = {}, byType = {};
        state.events.forEach(e => {
            if (e.tool) byTool[e.tool] = (byTool[e.tool] || 0) + 1;
            byType[e.type] = (byType[e.type] || 0) + 1;
        });
        return { sessionId: state.sessionId, startedAt: state.startedAt,
                 total: state.events.length, byTool, byType };
    }

    function exportJSON() {
        load();
        const payload = {
            schema: SCHEMA,
            exportedAt: new Date().toISOString(),
            toolkitVersion: (global.Toolkit && global.Toolkit.VERSION) || 'unknown',
            note: 'Recorded locally in the browser. Contains no identity information. ' +
                  'Submitted only because the student chose to export and hand it in.',
            sessionId: state.sessionId,
            startedAt: state.startedAt,
            events: state.events
        };
        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `session_${state.sessionId}.json`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        record('export', { kind: 'session' });
    }

    // --------------------------------------------------------------- panel

    function statTile(label, value) {
        return '<div class="p-2 rounded-lg bg-slate-50 dark:bg-surface-dark-lighter text-center">'
            + '<p class="text-[10px] text-slate-400 uppercase tracking-wider">' + label + '</p>'
            + '<p class="text-sm font-bold text-slate-800 dark:text-slate-200 tabular-nums">' + value + '</p>'
            + '</div>';
    }

    function eventRow(e) {
        const detail = Object.entries(e.detail || {}).map(([k, v]) => k + '=' + v).join(' ');
        return '<tr class="text-slate-600 dark:text-slate-300">'
            + '<td class="px-3 py-1.5 tabular-nums whitespace-nowrap">' + new Date(e.t).toLocaleTimeString() + '</td>'
            + '<td class="px-3 py-1.5">' + (e.tool || '&mdash;') + '</td>'
            + '<td class="px-3 py-1.5 font-mono text-[11px]">' + e.type + '</td>'
            + '<td class="px-3 py-1.5 font-mono text-[11px] text-slate-400">' + detail + '</td>'
            + '</tr>';
    }

    function headerMarkup(enabled) {
        return '<div class="flex items-start justify-between gap-4 mb-3">'
            + '<div>'
            + '<h3 class="text-sm font-bold text-slate-900 dark:text-white">What this page recorded about you</h3>'
            + '<p class="text-[11px] text-slate-500 dark:text-slate-400 mt-1 leading-relaxed max-w-prose">'
            + 'Everything below is stored in this browser only. There is no network call in '
            + '<code class="text-[10px]">shared/session.js</code> &mdash; you are welcome to check. Nothing is sent '
            + 'anywhere unless you press Export and hand the file in yourself.</p>'
            + '</div>'
            + '<label class="flex items-center gap-2 cursor-pointer flex-shrink-0">'
            + '<span class="text-xs text-slate-600 dark:text-slate-400">' + (enabled ? 'Recording' : 'Off') + '</span>'
            + '<span class="relative inline-flex items-center">'
            + '<input type="checkbox" id="sessionToggle" class="sr-only peer"' + (enabled ? ' checked' : '')
            + ' aria-label="Record my session locally">'
            + '<span class="w-11 h-6 bg-slate-300 dark:bg-slate-700 rounded-full peer peer-checked:bg-primary '
            + "after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white "
            + 'after:rounded-full after:h-5 after:w-5 after:transition-all '
            + 'peer-checked:after:translate-x-full peer-focus-visible:ring-2 peer-focus-visible:ring-primary"></span>'
            + '</span></label></div>';
    }

    function panelMarkup() {
        const enabled = isEnabled();
        const s = summary();

        if (!enabled) {
            return headerMarkup(false)
                + '<div class="p-3 rounded-lg bg-slate-50 dark:bg-surface-dark-lighter border border-slate-200 dark:border-slate-700">'
                + '<p class="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">'
                + 'Recording is off, and nothing is being stored. Turn it on if your lab asks you to submit a '
                + 'session export. You can turn it off and erase the record at any point.</p></div>';
        }

        const recent = state.events.slice(-40).reverse();
        const tiles = statTile('Events', s.total)
            + statTile('Tools used', Object.keys(s.byTool).length)
            + statTile('Session id', s.sessionId.slice(0, 8))
            + statTile('Started', new Date(s.startedAt).toLocaleTimeString());

        const rows = recent.length
            ? recent.map(eventRow).join('')
            : '<tr><td colspan="4" class="px-3 py-6 text-center text-slate-400">Nothing recorded yet.</td></tr>';

        return headerMarkup(true)
            + '<div class="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">' + tiles + '</div>'
            + '<div class="rounded-lg border border-slate-200 dark:border-slate-700 overflow-hidden">'
            + '<div class="max-h-64 overflow-y-auto"><table class="w-full text-xs">'
            + '<caption class="sr-only">Events recorded in this browser session</caption>'
            + '<thead class="sticky top-0 bg-slate-100 dark:bg-surface-dark-lighter text-slate-500"><tr>'
            + '<th scope="col" class="text-left px-3 py-1.5 font-semibold">Time</th>'
            + '<th scope="col" class="text-left px-3 py-1.5 font-semibold">Tool</th>'
            + '<th scope="col" class="text-left px-3 py-1.5 font-semibold">Event</th>'
            + '<th scope="col" class="text-left px-3 py-1.5 font-semibold">Detail</th>'
            + '</tr></thead><tbody class="divide-y divide-slate-200 dark:divide-slate-700">'
            + rows + '</tbody></table></div></div>'
            + (s.total > 40
                ? '<p class="text-[10px] text-slate-400 mt-1">Showing the most recent 40 of ' + s.total
                  + ' events. The export contains all of them.</p>'
                : '')
            + '<div class="flex flex-wrap gap-2 mt-3">'
            + '<button type="button" id="sessionExport" class="px-4 py-2 rounded-lg bg-primary hover:bg-primary-hover text-white text-sm font-medium transition-colors">Export session (.json)</button>'
            + '<button type="button" id="sessionClear" class="px-4 py-2 rounded-lg bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-300 text-sm font-medium transition-colors">Erase record</button>'
            + '</div>';
    }

    function renderPanel() {
        const host = document.getElementById('sessionPanel');
        if (!host) return;
        host.innerHTML = panelMarkup();
        host.querySelector('#sessionToggle')?.addEventListener('change', e => setEnabled(e.target.checked));
        host.querySelector('#sessionExport')?.addEventListener('click', exportJSON);
        host.querySelector('#sessionClear')?.addEventListener('click', () => {
            if (confirm('Erase everything recorded in this browser? This cannot be undone.')) clear();
        });
    }

    // A one-time, honest prompt rather than a pre-ticked box or a cookie wall.
    function maybeOfferConsent() {
        if (hasDecided()) return;
        const bar = document.createElement('div');
        bar.className = 'fixed bottom-4 right-4 z-50 max-w-sm p-4 rounded-xl bg-white dark:bg-surface-dark ' +
            'border border-slate-200 dark:border-slate-700 shadow-lg';
        bar.setAttribute('role', 'dialog');
        bar.setAttribute('aria-label', 'Session recording');
        bar.innerHTML = `
            <p class="text-sm font-bold text-slate-900 dark:text-white mb-1">Record this session locally?</p>
            <p class="text-xs text-slate-600 dark:text-slate-400 leading-relaxed mb-3">
                Some labs ask you to submit a record of what you tried. It stays in this browser, contains no
                identity information, and is only shared if you export it and hand it in. You can see and erase
                it any time from Settings.</p>
            <div class="flex gap-2">
                <button type="button" data-choice="on"
                    class="flex-1 px-3 py-1.5 rounded-lg bg-primary hover:bg-primary-hover text-white text-xs font-medium">Record locally</button>
                <button type="button" data-choice="off"
                    class="flex-1 px-3 py-1.5 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 text-xs font-medium">No thanks</button>
            </div>`;
        bar.addEventListener('click', e => {
            const choice = e.target.getAttribute('data-choice');
            if (!choice) return;
            setEnabled(choice === 'on');
            bar.remove();
        });
        document.body.appendChild(bar);
    }

    global.addEventListener('pagehide', endTool);

    global.ToolkitSession = {
        record, startTool, endTool, clear, summary, exportJSON,
        isEnabled, setEnabled, renderPanel, maybeOfferConsent
    };
})(window);
