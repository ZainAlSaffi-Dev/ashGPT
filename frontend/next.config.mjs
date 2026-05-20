/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // ``typedRoutes`` was promoted out of ``experimental`` in Next 15.4.
  typedRoutes: true,
  async rewrites() {
    const api = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';
    return [
      { source: '/api/:path*', destination: `${api}/:path*` },
    ];
  },
};

export default nextConfig;
