let githubStarsCache = null;

async function fetchGithubStars() {
  if (githubStarsCache != null) return githubStarsCache;

  const res = await fetch("https://api.github.com/repos/geniex-ai/geniex-sdk");
  const data = await res.json();

  githubStarsCache = data;
  return githubStarsCache;
}

async function updateGithubStars() {
  const data = await fetchGithubStars();
  const svg = document.querySelector(
    'svg[style*="github.svg"]'
  );

  if (!svg) return;

  let textNode = svg.nextSibling;

  while (textNode && textNode.nodeType === Node.TEXT_NODE && !textNode.textContent.trim()) {
    textNode = textNode.nextSibling;
  }

  if (textNode && textNode.nodeType === Node.TEXT_NODE) {
    textNode.textContent = ` ${formatStars(data.stargazers_count)}`;
  }
}

function formatStars(num) {
  if (num >= 1000) {
    return (num / 1000).toFixed(1) + "K";
  }
  return String(num);
}

updateGithubStars();

if (!window.__githubStarsInterval) {
  window.__githubStarsInterval = setInterval(() => {
    updateGithubStars();
  }, 1000);
}
