import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Structure content lives in ../content, outside the Vite root.
  server: { fs: { allow: ['..'] } },
})
