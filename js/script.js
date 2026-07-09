// ===== scroll progress bar =====
const progressBar = document.getElementById('progressBar');
function updateProgress() {
  const scrollTop = window.scrollY;
  const docHeight = document.documentElement.scrollHeight - window.innerHeight;
  const pct = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
  progressBar.style.width = pct + '%';
}

// ===== navbar scrolled state =====
const navbar = document.getElementById('navbar');
function updateNavbar() {
  navbar.classList.toggle('scrolled', window.scrollY > 20);
}

window.addEventListener('scroll', () => {
  updateProgress();
  updateNavbar();
  updateActiveNav();
}, { passive: true });

// ===== mobile nav toggle =====
const navToggle = document.getElementById('navToggle');
const navLinks = document.getElementById('navLinks');
navToggle.addEventListener('click', () => {
  navToggle.classList.toggle('open');
  navLinks.classList.toggle('open');
});
navLinks.querySelectorAll('a').forEach(link => {
  link.addEventListener('click', () => {
    navToggle.classList.remove('open');
    navLinks.classList.remove('open');
  });
});

// ===== active nav link on scroll =====
const sections = document.querySelectorAll('main .section, .footer');
const navAnchors = document.querySelectorAll('.nav-links a[data-nav]');
function updateActiveNav() {
  let currentId = '';
  sections.forEach(sec => {
    const rect = sec.getBoundingClientRect();
    if (rect.top <= 120 && rect.bottom >= 120) {
      currentId = sec.id;
    }
  });
  navAnchors.forEach(a => {
    a.classList.toggle('active', a.getAttribute('href') === '#' + currentId);
  });
}

// ===== reveal-on-scroll animation =====
const revealEls = document.querySelectorAll('.reveal');
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('in-view');
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
revealEls.forEach(el => revealObserver.observe(el));

// ===== expandable project cards =====
document.querySelectorAll('.project-head[data-toggle]').forEach(head => {
  head.addEventListener('click', () => {
    const project = head.closest('.project');
    const isOpen = project.classList.contains('open');
    // close other cards within the same list for a cleaner accordion feel
    const list = project.closest('.project-list');
    list.querySelectorAll('.project.open').forEach(p => {
      if (p !== project) p.classList.remove('open');
    });
    project.classList.toggle('open', !isOpen);

    if (!isOpen) {
      const pdfEmbed = project.querySelector('.pdf-embed[data-pdf]');
      if (pdfEmbed && !pdfEmbed.dataset.loaded) {
        loadPdfEmbed(pdfEmbed);
      }
    }
  });
});

function loadPdfEmbed(pdfEmbed) {
  pdfEmbed.dataset.loaded = 'true';
  const src = pdfEmbed.dataset.pdf;
  const iframe = document.createElement('iframe');
  iframe.src = src + '#toolbar=1&view=FitH';
  iframe.title = 'PDF 미리보기';
  iframe.loading = 'lazy';
  const fallback = document.createElement('a');
  fallback.href = src;
  fallback.target = '_blank';
  fallback.rel = 'noopener';
  fallback.className = 'pdf-fallback-link';
  fallback.textContent = '화면에 잘 보이지 않으면 새 탭에서 크게 보기 →';
  pdfEmbed.innerHTML = '';
  pdfEmbed.appendChild(iframe);
  pdfEmbed.appendChild(fallback);
}

// ===== inline code viewer =====
const escapeHtml = (str) => str
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;');

function renderArticleText(text) {
  const blocks = text.split(/\n\s*\n/).map(b => b.trim()).filter(Boolean);
  return blocks.map(b => {
    if (b.startsWith('## ')) {
      return `<h4 class="article-heading">${escapeHtml(b.slice(3))}</h4>`;
    }
    return `<p>${escapeHtml(b)}</p>`;
  }).join('');
}

document.querySelectorAll('.code-toggle').forEach(btn => {
  btn.addEventListener('click', async () => {
    const panel = btn.closest('.project-detail').querySelector('.code-panel');
    const alreadyOpen = btn.classList.contains('open');
    const isArticle = !!btn.dataset.article;
    const src = btn.dataset.code || btn.dataset.article;

    // reset sibling toggle buttons sharing this panel
    btn.closest('.project-links').querySelectorAll('.code-toggle').forEach(b => b.classList.remove('open'));

    if (alreadyOpen) {
      panel.innerHTML = '';
      return;
    }

    btn.classList.add('open');
    panel.innerHTML = `<p class="code-panel-loading">${isArticle ? '아티클 불러오는 중…' : '코드 불러오는 중…'}</p>`;

    try {
      const res = await fetch(src);
      const text = await res.text();
      if (isArticle) {
        panel.innerHTML = `<div class="article-panel">${renderArticleText(text)}</div>`;
      } else {
        const lang = btn.dataset.lang || 'python';
        const fileName = src.split('/').pop();
        panel.innerHTML = `<p class="code-panel-name">${fileName}</p><pre class="line-numbers"><code class="language-${lang}">${escapeHtml(text)}</code></pre>`;
        if (window.Prism) {
          Prism.highlightAllUnder(panel);
        }
      }
    } catch (err) {
      panel.innerHTML = `<p class="code-panel-loading">${isArticle ? '아티클을' : '코드를'} 불러오지 못했습니다.</p>`;
    }
  });
});

// init
updateProgress();
updateNavbar();
updateActiveNav();
