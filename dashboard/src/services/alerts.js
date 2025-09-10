import API from "./api";

export const getAlerts = () => API.get("/alerts");
