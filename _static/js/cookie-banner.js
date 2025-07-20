var cookieBanner = {
  init: function() {
    cookieBanner.bind();

    var cookieExists = cookieBanner.cookieExists();

    if (!cookieExists) {
      cookieBanner.showCookieNotice();
    }
  },

  bind: function() {
    $(".close-button").on("click", function() {
      cookieBanner.hideCookieNotice();
      cookieBanner.setCookie(); // Set the cookie or local storage item when the banner is closed
    });
  },

  cookieExists: function() {
    return localStorage.getItem("returningPytorchUser") !== null;
  },

  setCookie: function() {
    localStorage.setItem("returningPytorchUser", true);
  },

  showCookieNotice: function() {
    $(".cookie-banner-wrapper").addClass("is-visible");
  },

  hideCookieNotice: function() {
    $(".cookie-banner-wrapper").removeClass("is-visible");
  }
};

$(function() {
  cookieBanner.init();
});
