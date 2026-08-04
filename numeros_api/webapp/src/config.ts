// Único lugar donde se lee la variable de entorno
export const API_URL = import.meta.env.PUBLIC_API_URL ?? 'http://localhost:8000';