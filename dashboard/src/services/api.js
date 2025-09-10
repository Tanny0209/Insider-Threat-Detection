import axios from "axios";

// Base API instance
const API = axios.create({
  baseURL: "http://localhost:8000", // backend Flask/FastAPI server
});

export default API;
