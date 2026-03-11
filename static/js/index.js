$(document).ready(function() {
  $(".navbar-burger").click(function() {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  const featureSlides = $(".feature-slide");
  const featureIndicators = $(".feature-indicator");
  let featureIndex = 0;

  function setFeature(index) {
    if (!featureSlides.length) {
      return;
    }
    featureIndex = (index + featureSlides.length) % featureSlides.length;
    featureSlides.removeClass("is-active").eq(featureIndex).addClass("is-active");
    featureIndicators.removeClass("is-active").eq(featureIndex).addClass("is-active");
  }

  $(".feature-nav.prev").click(function() {
    setFeature(featureIndex - 1);
  });

  $(".feature-nav.next").click(function() {
    setFeature(featureIndex + 1);
  });

  featureIndicators.each(function(index) {
    $(this).click(function() {
      setFeature(index);
    });
  });
});
