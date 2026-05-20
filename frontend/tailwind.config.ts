import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./src/**/*.{ts,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        // Calm law-library palette: cream + ink + a single accent.
        ink: {
          DEFAULT: '#1a1d24',
          muted: '#4a5060',
          soft: '#8a8f9e',
        },
        parchment: {
          DEFAULT: '#fbf9f4',
          warm: '#f4efe5',
        },
        accent: {
          DEFAULT: '#7a3b2e', // burgundy
          hover: '#9a4c3b',
        },
      },
      fontFamily: {
        serif: ['Georgia', 'Charter', 'Iowan Old Style', 'serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
};

export default config;
