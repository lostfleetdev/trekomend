// @ts-check
import { defineConfig } from 'astro/config';

export default defineConfig({
    output: 'static',
    site: 'https://trekomend.chaospunk.space',
    prefetch: {
        prefetchAll: true,
        defaultStrategy: 'viewport',
    },
    vite: {
        server: {
            proxy: {
                '/api': {
                    target: 'http://127.0.0.1:6767',
                    changeOrigin: true,
                },
            },
        },
    },
});
