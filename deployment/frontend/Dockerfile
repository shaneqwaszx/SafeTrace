FROM node:20-alpine AS build
WORKDIR /app
COPY frontend-react/package.json frontend-react/package-lock.json ./
RUN npm ci
COPY frontend-react ./
ARG VITE_SAFETRACE_API_BASE_URL
ENV VITE_SAFETRACE_API_BASE_URL=$VITE_SAFETRACE_API_BASE_URL
RUN test -n "$VITE_SAFETRACE_API_BASE_URL" && npm run build

FROM nginx:1.27-alpine
COPY deployment/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD wget -q -O /dev/null http://127.0.0.1:8080/
