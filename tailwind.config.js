/** Tailwind build for the AI Ethics Toolkit.
 *
 *  Replaces the Play CDN (cdn.tailwindcss.com), which Tailwind documents as
 *  not for production: it shipped a ~400 KB compiler to every visitor and
 *  rebuilt the stylesheet on every page load.
 *
 *  Build with:  npm run build:css
 */
module.exports = {
    darkMode: 'class',
    content: [
        './*.html',
        './tools/*.html',
        './shared/*.js',
    ],
    // Colour utilities assembled at runtime from data (tool cards, risk
    // levels, framework colours) are invisible to the content scanner, so
    // they are listed explicitly. Dropping this breaks them silently.
    safelist: [
        {
            pattern: /^(bg|text|border)-(blue|purple|orange|red|teal|indigo|pink|cyan|emerald|green|yellow|slate|amber)-(400|500|600)$/,
            variants: ['dark', 'hover', 'dark:hover'],
        },
        {
            pattern: /^bg-(blue|purple|orange|red|teal|indigo|pink|cyan|emerald|green|yellow|amber)-500\/(5|10|20|30)$/,
        },
        {
            pattern: /^border-(green|red|yellow|orange|purple|amber|slate)-500\/(20|30|40)$/,
        },
    ],
    theme: {
        extend: {
            colors: {
                primary: '#137fec',
                'primary-hover': '#0f6bd1',
                'primary-light': '#4da3f5',
                'background-light': '#f6f7f8',
                'background-dark': '#101922',
                'surface-dark': '#182430',
                'surface-dark-lighter': '#212e3b',
                danger: '#ef4444',
                success: '#22c55e',
                warning: '#f59e0b',
            },
            fontFamily: {
                display: ['Space Grotesk', 'ui-sans-serif', 'system-ui', 'sans-serif'],
            },
            borderRadius: {
                DEFAULT: '0.25rem',
                lg: '0.5rem',
                xl: '0.75rem',
                '2xl': '1rem',
                full: '9999px',
            },
        },
    },
    plugins: [
        require('@tailwindcss/forms'),
        require('@tailwindcss/container-queries'),
    ],
};
