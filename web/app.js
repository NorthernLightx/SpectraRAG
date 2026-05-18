const themeBtn = document.getElementById("themeToggle");

const savedTheme = localStorage.getItem("rag.theme");
const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
document.documentElement.setAttribute(
  "data-theme",
  savedTheme || (prefersLight ? "light" : "dark")
);
themeBtn.addEventListener("click", () => {
  const next =
    document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("rag.theme", next);
});
