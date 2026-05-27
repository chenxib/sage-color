document.addEventListener("DOMContentLoaded", function () {
  const carousel = document.querySelector("[data-gallery-carousel]");
  if (!carousel) {
    return;
  }

  const track = carousel.querySelector("[data-gallery-track]");
  const viewport = carousel.querySelector("[data-gallery-viewport]");
  const slides = Array.from(carousel.querySelectorAll(".gallery-slide"));
  const previous = document.querySelector("[data-gallery-prev]");
  const next = document.querySelector("[data-gallery-next]");
  const dotsContainer = document.querySelector("[data-gallery-dots]");

  if (!track || !viewport || !slides.length || !previous || !next || !dotsContainer) {
    return;
  }

  let index = 0;
  let touchStartX = 0;
  let touchDeltaX = 0;

  const dots = slides.map(function (_, dotIndex) {
    const dot = document.createElement("button");
    dot.type = "button";
    dot.className = "gallery-dot";
    dot.setAttribute("aria-label", "Show example " + (dotIndex + 1));
    dot.addEventListener("click", function () {
      render(dotIndex);
    });
    dotsContainer.appendChild(dot);
    return dot;
  });

  function render(nextIndex) {
    index = (nextIndex + slides.length) % slides.length;
    track.style.transform = "translateX(-" + index * 100 + "%)";

    slides.forEach(function (slide, slideIndex) {
      slide.setAttribute("aria-hidden", String(slideIndex !== index));
    });

    dots.forEach(function (dot, dotIndex) {
      const active = dotIndex === index;
      dot.classList.toggle("is-active", active);
      dot.setAttribute("aria-current", String(active));
    });
  }

  previous.addEventListener("click", function () {
    render(index - 1);
  });

  next.addEventListener("click", function () {
    render(index + 1);
  });

  carousel.addEventListener("keydown", function (event) {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      render(index - 1);
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      render(index + 1);
    }
  });

  viewport.addEventListener(
    "touchstart",
    function (event) {
      touchStartX = event.touches[0].clientX;
      touchDeltaX = 0;
    },
    { passive: true }
  );

  viewport.addEventListener(
    "touchmove",
    function (event) {
      touchDeltaX = event.touches[0].clientX - touchStartX;
    },
    { passive: true }
  );

  viewport.addEventListener("touchend", function () {
    if (Math.abs(touchDeltaX) > 50) {
      render(index + (touchDeltaX < 0 ? 1 : -1));
    }
  });

  render(0);
});
