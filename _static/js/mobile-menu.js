$(document).ready(function() {
  mobileMenu.bind();
});

window.mobileMenu = {
  bind: function() {
    $("[data-behavior='open-mobile-menu']").on('click', function(e) {
      e.preventDefault();
      e.stopPropagation(); // Add this line

      // If menu is already open, close it
      if ($(".mobile-main-menu").hasClass("open")) {
        mobileMenu.close();
        return;
      }

      $(".mobile-main-menu").addClass("open");
      $("body").addClass('no-scroll');

      mobileMenu.listenForResize();

      $(document).on('click.ForMobileMenu', function(event) {
        if (!$(event.target).closest('.mobile-main-menu-links-container').length &&
            !$(event.target).is('[data-behavior="open-mobile-menu"]') &&
            !$(event.target).closest('[data-behavior="open-mobile-menu"]').length) {
          mobileMenu.close();
        }
      });
    });

    $("[data-behavior='close-mobile-menu']").on('click', function(e) {
      e.preventDefault();
      mobileMenu.close();
    });
  },

  listenForResize: function() {
    $(window).on('resize.ForMobileMenu', function() {
      if ($(this).width() > 959) {
        mobileMenu.close();
      }
    });
  },

  close: function() {
    $(".mobile-main-menu").removeClass("open");
    $("body").removeClass('no-scroll');
    $(window).off('resize.ForMobileMenu');
    $(document).off('click.ForMobileMenu');
  }
};
