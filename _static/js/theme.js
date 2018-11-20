require=(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
window.utilities = {
  scrollTop: function() {
    var supportPageOffset = window.pageXOffset !== undefined;
    var isCSS1Compat = ((document.compatMode || "") === "CSS1Compat");
    var scrollLeft = supportPageOffset ? window.pageXOffset : isCSS1Compat ? document.documentElement.scrollLeft : document.body.scrollLeft;
    return supportPageOffset ? window.pageYOffset : isCSS1Compat ? document.documentElement.scrollTop : document.body.scrollTop;
  },

  // Modified from https://stackoverflow.com/a/27078401
  throttle: function(func, wait, options) {
    var context, args, result;
    var timeout = null;
    var previous = 0;
    if (!options) options = {};
    var later = function() {
      previous = options.leading === false ? 0 : Date.now();
      timeout = null;
      result = func.apply(context, args);
      if (!timeout) context = args = null;
    };
    return function() {
      var now = Date.now();
      if (!previous && options.leading === false) previous = now;
      var remaining = wait - (now - previous);
      context = this;
      args = arguments;
      if (remaining <= 0 || remaining > wait) {
        if (timeout) {
          clearTimeout(timeout);
          timeout = null;
        }
        previous = now;
        result = func.apply(context, args);
        if (!timeout) context = args = null;
      } else if (!timeout && options.trailing !== false) {
        timeout = setTimeout(later, remaining);
      }
      return result;
    };
  },

  closest: function (el, selector) {
    var matchesFn;

    // find vendor prefix
    ['matches','webkitMatchesSelector','mozMatchesSelector','msMatchesSelector','oMatchesSelector'].some(function(fn) {
      if (typeof document.body[fn] == 'function') {
        matchesFn = fn;
        return true;
      }
      return false;
    });

    var parent;

    // traverse parents
    while (el) {
      parent = el.parentElement;
      if (parent && parent[matchesFn](selector)) {
        return parent;
      }
      el = parent;
    }

    return null;
  },

  // Modified from https://stackoverflow.com/a/18953277
  offset: function(elem) {
    if (!elem) {
      return;
    }

    rect = elem.getBoundingClientRect();

    // Make sure element is not hidden (display: none) or disconnected
    if (rect.width || rect.height || elem.getClientRects().length) {
      var doc = elem.ownerDocument;
      var docElem = doc.documentElement;

      return {
        top: rect.top + window.pageYOffset - docElem.clientTop,
        left: rect.left + window.pageXOffset - docElem.clientLeft
      };
    }
  },

  headersHeight: function() {
    return document.getElementById("header-holder").offsetHeight +
           document.getElementById("pytorch-page-level-bar").offsetHeight;
  }
}

},{}],2:[function(require,module,exports){
// Modified from https://stackoverflow.com/a/32396543
window.highlightNavigation = {
  navigationListItems: document.querySelectorAll("#pytorch-right-menu li"),
  sections: document.querySelectorAll(".pytorch-article .section"),
  sectionIdTonavigationLink: {},

  bind: function() {
    if (!sideMenus.displayRightMenu) {
      return;
    };

    for (var i = 0; i < highlightNavigation.sections.length; i++) {
      var id = highlightNavigation.sections[i].id;
      highlightNavigation.sectionIdTonavigationLink[id] =
        document.querySelectorAll('#pytorch-right-menu li a[href="#' + id + '"]')[0];
    }

    $(window).scroll(utilities.throttle(highlightNavigation.highlight, 100));
  },

  highlight: function() {
    var rightMenu = document.getElementById("pytorch-right-menu");

    // If right menu is not on the screen don't bother
    if (rightMenu.offsetWidth === 0 && rightMenu.offsetHeight === 0) {
      return;
    }

    var scrollPosition = utilities.scrollTop();
    var OFFSET_TOP_PADDING = 25;
    var offset = document.getElementById("header-holder").offsetHeight +
                 document.getElementById("pytorch-page-level-bar").offsetHeight +
                 OFFSET_TOP_PADDING;

    var sections = highlightNavigation.sections;

    for (var i = (sections.length - 1); i >= 0; i--) {
      var currentSection = sections[i];
      var sectionTop = utilities.offset(currentSection).top;

      if (scrollPosition >= sectionTop - offset) {
        var navigationLink = highlightNavigation.sectionIdTonavigationLink[currentSection.id];
        var navigationListItem = utilities.closest(navigationLink, "li");

        if (navigationListItem && !navigationListItem.classList.contains("active")) {
          for (var i = 0; i < highlightNavigation.navigationListItems.length; i++) {
            var el = highlightNavigation.navigationListItems[i];
            if (el.classList.contains("active")) {
              el.classList.remove("active");
            }
          }

          navigationListItem.classList.add("active");

          // Scroll to active item. Not a requested feature but we could revive it. Needs work.

          // var menuTop = $("#pytorch-right-menu").position().top;
          // var itemTop = navigationListItem.getBoundingClientRect().top;
          // var TOP_PADDING = 20
          // var newActiveTop = $("#pytorch-side-scroll-right").scrollTop() + itemTop - menuTop - TOP_PADDING;

          // $("#pytorch-side-scroll-right").animate({
          //   scrollTop: newActiveTop
          // }, 100);
        }

        break;
      }
    }
  }
};

},{}],3:[function(require,module,exports){
MathJax.Hub.Config({
    messageStyle: "none",
    scale: 100,
    "HTML-CSS": {
        showMathMenu: false,
        linebreaks: { automatic: true, width: "container" } ,
        preferredFont: "STIX",
        availableFonts: ["STIX","TeX"]
    },
    SVG: { linebreaks: { automatic: true, width: "container" } }
 });

 MathJax.Hub.Configured();

},{}],4:[function(require,module,exports){
window.mobileMenu = {
  bind: function() {
    $("[data-behavior='open-mobile-menu']").on('click', function(e) {
      e.preventDefault();
      $(".mobile-main-menu").addClass("open");
      $("body").addClass('no-scroll');

      mobileMenu.listenForResize();
    });

    $("[data-behavior='close-mobile-menu']").on('click', function(e) {
      e.preventDefault();
      mobileMenu.close();
    });
  },

  listenForResize: function() {
    $(window).on('resize.ForMobileMenu', function() {
      if ($(this).width() > 768) {
        mobileMenu.close();
      }
    });
  },

  close: function() {
    $(".mobile-main-menu").removeClass("open");
    $("body").removeClass('no-scroll');
    $(window).off('resize.ForMobileMenu');
  }
};

},{}],5:[function(require,module,exports){
window.mobileTOC = {
  bind: function() {
    $("[data-behavior='toggle-table-of-contents']").on("click", function(e) {
      e.preventDefault();

      var $parent = $(this).parent();

      if ($parent.hasClass("is-open")) {
        $parent.removeClass("is-open");
        $(".pytorch-left-menu").slideUp(200, function() {
          $(this).css({display: ""});
        });
      } else {
        $parent.addClass("is-open");
        $(".pytorch-left-menu").slideDown(200);
      }
    });
  }
}

},{}],6:[function(require,module,exports){
window.pytorchAnchors = {
  bind: function() {
    // Replace Sphinx-generated anchors with anchorjs ones
    $(".headerlink").text("");

    window.anchors.add(".pytorch-article .headerlink");

    $(".anchorjs-link").each(function() {
      var $headerLink = $(this).closest(".headerlink");
      var href = $headerLink.attr("href");
      var clone = this.outerHTML;

      $clone = $(clone).attr("href", href);
      $headerLink.before($clone);
      $headerLink.remove();
    });
  }
};

},{}],7:[function(require,module,exports){
// Modified from https://stackoverflow.com/a/13067009
// Going for a JS solution to scrolling to an anchor so we can benefit from
// less hacky css and smooth scrolling.

window.scrollToAnchor = {
  bind: function() {
    var document = window.document;
    var history = window.history;
    var location = window.location
    var HISTORY_SUPPORT = !!(history && history.pushState);

    var anchorScrolls = {
      ANCHOR_REGEX: /^#[^ ]+$/,
      offsetHeightPx: function() {
        var OFFSET_HEIGHT_PADDING = 20;
        return document.getElementById("header-holder").offsetHeight +
               document.getElementById("pytorch-page-level-bar").offsetHeight +
               OFFSET_HEIGHT_PADDING;
      },

      /**
       * Establish events, and fix initial scroll position if a hash is provided.
       */
      init: function() {
        this.scrollToCurrent();
        // This interferes with clicks below it, causing a double fire
        // $(window).on('hashchange', $.proxy(this, 'scrollToCurrent'));
        $('body').on('click', 'a', $.proxy(this, 'delegateAnchors'));
        $('body').on('click', '#pytorch-right-menu li span', $.proxy(this, 'delegateSpans'));
      },

      /**
       * Return the offset amount to deduct from the normal scroll position.
       * Modify as appropriate to allow for dynamic calculations
       */
      getFixedOffset: function() {
        return this.offsetHeightPx();
      },

      /**
       * If the provided href is an anchor which resolves to an element on the
       * page, scroll to it.
       * @param  {String} href
       * @return {Boolean} - Was the href an anchor.
       */
      scrollIfAnchor: function(href, pushToHistory) {
        var match, anchorOffset;

        if(!this.ANCHOR_REGEX.test(href)) {
          return false;
        }

        match = document.getElementById(href.slice(1));

        if(match) {
          var anchorOffset = $(match).offset().top - this.getFixedOffset();

          $('html, body').scrollTop(anchorOffset);

          // Add the state to history as-per normal anchor links
          if(HISTORY_SUPPORT && pushToHistory) {
            history.pushState({}, document.title, location.pathname + href);
          }
        }

        return !!match;
      },

      /**
       * Attempt to scroll to the current location's hash.
       */
      scrollToCurrent: function(e) {
        if(this.scrollIfAnchor(window.location.hash) && e) {
          e.preventDefault();
        }
      },

      delegateSpans: function(e) {
        var elem = utilities.closest(e.target, "a");

        if(this.scrollIfAnchor(elem.getAttribute('href'), true)) {
          e.preventDefault();
        }
      },

      /**
       * If the click event's target was an anchor, fix the scroll position.
       */
      delegateAnchors: function(e) {
        var elem = e.target;

        if(this.scrollIfAnchor(elem.getAttribute('href'), true)) {
          e.preventDefault();
        }
      }
    };

    $(document).ready($.proxy(anchorScrolls, 'init'));
  }
};

},{}],8:[function(require,module,exports){
window.sideMenus = {
  displayRightMenu: document.querySelectorAll("#pytorch-right-menu li").length > 1,

  isFixedToBottom: false,

  bind: function() {
    sideMenus.handleLeftMenu();

    if (sideMenus.displayRightMenu) {
      // Show the right menu container
      document.getElementById("pytorch-content-right").classList.add("show");

      // Don't show the Shortcuts menu title text unless there are menu items
      document.getElementById("pytorch-shortcuts-wrapper").style.display = "block";

      // Remove superfluous titles unless there are more than one
      var titles = document.querySelectorAll("#pytorch-side-scroll-right > ul > li");

      if (titles.length === 1) {
        titles[0].querySelector("a.reference.internal").style.display = "none";
      }

      // Start the Shortcuts menu at the article's H1 position
      document.getElementById("pytorch-right-menu").style["margin-top"] = sideMenus.rightMenuInitialTop() + "px";

      sideMenus.handleRightMenu();
    }

    $(window).on('resize scroll', function(e) {
      sideMenus.handleLeftMenu();

      if (sideMenus.displayRightMenu) {
        sideMenus.handleRightMenu();
      }
    });
  },

  rightMenuInitialTop: function() {
    return utilities.headersHeight();
  },

  handleLeftMenu: function () {
    var windowHeight = window.innerHeight;
    var topOfFooterRelativeToWindow = document.getElementById("docs-tutorials-resources").getBoundingClientRect().top;

    if (topOfFooterRelativeToWindow >= windowHeight) {
      document.getElementById("pytorch-left-menu").style.height = "100%";
    } else {
      var howManyPixelsOfTheFooterAreInTheWindow = windowHeight - topOfFooterRelativeToWindow;
      var headerHeight = document.getElementById('header-holder').offsetHeight;
      var leftMenuDifference = howManyPixelsOfTheFooterAreInTheWindow + headerHeight;

      document.getElementById("pytorch-left-menu").style.height = (windowHeight - leftMenuDifference) + "px";
    }
  },

  handleRightMenu: function() {
    var rightMenu = document.getElementById("pytorch-right-menu");
    var scrollPos = utilities.scrollTop();

    if (scrollPos === 0) {
      rightMenu.style["margin-top"] = sideMenus.rightMenuInitialTop() + "px";
      return;
    }

    var rightMenuList = rightMenu.getElementsByTagName("ul")[0];
    var rightMenuBottom = utilities.offset(rightMenuList).top + rightMenuList.offsetHeight;
    var footerTop = utilities.offset(document.getElementById("docs-tutorials-resources")).top;
    var isBottomOfMenuPastOrCloseToFooter = rightMenuBottom >= footerTop  || footerTop - rightMenuBottom <= 40
    var heightOfFooterOnScreen = $(window).height() - document.getElementById("docs-tutorials-resources").getBoundingClientRect().top;

    if (heightOfFooterOnScreen < 0) {
      heightOfFooterOnScreen = 0;
    }

    // If the right menu is already fixed to the bottom
    if (this.isFixedToBottom) {
      var isFooterOnScreen = isElementInViewport(document.getElementById("docs-tutorials-resources"));

      // If the footer is still on the screen, we want to keep the menu where it is
      if (isFooterOnScreen) {
        bottom = heightOfFooterOnScreen;
        rightMenu.style["margin-top"] = "auto";
        rightMenu.style.bottom = bottom + "px";
      } else {
        // If the footer is not on the screen, we want to break the side menu out of the bottom
        this.isFixedToBottom = false;
        rightMenu.style.height = (window.innerHeight - heightOfFooterOnScreen - utilities.headersHeight()) + "px";
        rightMenu.style["margin-top"] = sideMenus.rightMenuInitialTop() + "px";
        rightMenu.style.bottom = bottom;
      }

    // If the side menu is past the footer's top or close to it (by 40 pixels)
    // we fix the menu to the bottom
    } else if (isBottomOfMenuPastOrCloseToFooter) {
      var isFooterOnScreen = isElementInViewport(document.getElementById("docs-tutorials-resources"));
      var bottom = 0;

      this.isFixedToBottom = true;

      if (isFooterOnScreen) {
        bottom = heightOfFooterOnScreen;
        rightMenu.style["margin-top"] = "auto";
        rightMenu.style.bottom = bottom + "px";
      } else {
        rightMenu.style.height = (window.innerHeight - heightOfFooterOnScreen - utilities.headersHeight()) + "px";
        rightMenu.style["margin-top"] = sideMenus.rightMenuInitialTop() + "px";
        rightMenu.style.bottom = bottom;
      }
    } else {
      this.isFixedToBottom = false;
      rightMenu.style.height = (window.innerHeight - heightOfFooterOnScreen - utilities.headersHeight()) + "px";
      rightMenu.style["margin-top"] = sideMenus.rightMenuInitialTop() + "px";
      rightMenu.style.bottom = bottom;
    }
  }
};

function isElementInViewport(el) {
  var rect = el.getBoundingClientRect();

  return rect.bottom > 0 &&
    rect.right > 0 &&
    rect.left < (window.innerWidth || document.documentElement.clientWidth) &&
    rect.top < (window.innerHeight || document.documentElement.clientHeight);
}

},{}],"pytorch-sphinx-theme":[function(require,module,exports){
var jQuery = (typeof(window) != 'undefined') ? window.jQuery : require('jquery');

// Sphinx theme nav state
function ThemeNav () {

    var nav = {
        navBar: null,
        win: null,
        winScroll: false,
        winResize: false,
        linkScroll: false,
        winPosition: 0,
        winHeight: null,
        docHeight: null,
        isRunning: false
    };

    nav.enable = function (withStickyNav) {
        var self = this;

        // TODO this can likely be removed once the theme javascript is broken
        // out from the RTD assets. This just ensures old projects that are
        // calling `enable()` get the sticky menu on by default. All other cals
        // to `enable` should include an argument for enabling the sticky menu.
        if (typeof(withStickyNav) == 'undefined') {
            withStickyNav = true;
        }

        if (self.isRunning) {
            // Only allow enabling nav logic once
            return;
        }

        self.isRunning = true;
        jQuery(function ($) {
            self.init($);

            self.reset();
            self.win.on('hashchange', self.reset);

            if (withStickyNav) {
                // Set scroll monitor
                self.win.on('scroll', function () {
                    if (!self.linkScroll) {
                        if (!self.winScroll) {
                            self.winScroll = true;
                            requestAnimationFrame(function() { self.onScroll(); });
                        }
                    }
                });
            }

            // Set resize monitor
            self.win.on('resize', function () {
                if (!self.winResize) {
                    self.winResize = true;
                    requestAnimationFrame(function() { self.onResize(); });
                }
            });

            self.onResize();
        });

    };

    // TODO remove this with a split in theme and Read the Docs JS logic as
    // well, it's only here to support 0.3.0 installs of our theme.
    nav.enableSticky = function() {
        this.enable(true);
    };

    nav.init = function ($) {
        var doc = $(document),
            self = this;

        this.navBar = $('div.pytorch-side-scroll:first');
        this.win = $(window);

        // Set up javascript UX bits
        $(document)
            // Shift nav in mobile when clicking the menu.
            .on('click', "[data-toggle='pytorch-left-menu-nav-top']", function() {
                $("[data-toggle='wy-nav-shift']").toggleClass("shift");
                $("[data-toggle='rst-versions']").toggleClass("shift");
            })

            // Nav menu link click operations
            .on('click', ".pytorch-menu-vertical .current ul li a", function() {
                var target = $(this);
                // Close menu when you click a link.
                $("[data-toggle='wy-nav-shift']").removeClass("shift");
                $("[data-toggle='rst-versions']").toggleClass("shift");
                // Handle dynamic display of l3 and l4 nav lists
                self.toggleCurrent(target);
                self.hashChange();
            })
            .on('click', "[data-toggle='rst-current-version']", function() {
                $("[data-toggle='rst-versions']").toggleClass("shift-up");
            })

        // Make tables responsive
        $("table.docutils:not(.field-list,.footnote,.citation)")
            .wrap("<div class='wy-table-responsive'></div>");

        // Add extra class to responsive tables that contain
        // footnotes or citations so that we can target them for styling
        $("table.docutils.footnote")
            .wrap("<div class='wy-table-responsive footnote'></div>");
        $("table.docutils.citation")
            .wrap("<div class='wy-table-responsive citation'></div>");

        // Add expand links to all parents of nested ul
        $('.pytorch-menu-vertical ul').not('.simple').siblings('a').each(function () {
            var link = $(this);
                expand = $('<span class="toctree-expand"></span>');
            expand.on('click', function (ev) {
                self.toggleCurrent(link);
                ev.stopPropagation();
                return false;
            });
            link.prepend(expand);
        });
    };

    nav.reset = function () {
        // Get anchor from URL and open up nested nav
        var anchor = encodeURI(window.location.hash) || '#';

        try {
            var vmenu = $('.pytorch-menu-vertical');
            var link = vmenu.find('[href="' + anchor + '"]');
            if (link.length === 0) {
                // this link was not found in the sidebar.
                // Find associated id element, then its closest section
                // in the document and try with that one.
                var id_elt = $('.document [id="' + anchor.substring(1) + '"]');
                var closest_section = id_elt.closest('div.section');
                link = vmenu.find('[href="#' + closest_section.attr("id") + '"]');
                if (link.length === 0) {
                    // still not found in the sidebar. fall back to main section
                    link = vmenu.find('[href="#"]');
                }
            }
            // If we found a matching link then reset current and re-apply
            // otherwise retain the existing match
            if (link.length > 0) {
                $('.pytorch-menu-vertical .current').removeClass('current');
                link.addClass('current');
                link.closest('li.toctree-l1').addClass('current');
                link.closest('li.toctree-l1').parent().addClass('current');
                link.closest('li.toctree-l1').addClass('current');
                link.closest('li.toctree-l2').addClass('current');
                link.closest('li.toctree-l3').addClass('current');
                link.closest('li.toctree-l4').addClass('current');
            }
        }
        catch (err) {
            console.log("Error expanding nav for anchor", err);
        }

    };

    nav.onScroll = function () {
        this.winScroll = false;
        var newWinPosition = this.win.scrollTop(),
            winBottom = newWinPosition + this.winHeight,
            navPosition = this.navBar.scrollTop(),
            newNavPosition = navPosition + (newWinPosition - this.winPosition);
        if (newWinPosition < 0 || winBottom > this.docHeight) {
            return;
        }
        this.navBar.scrollTop(newNavPosition);
        this.winPosition = newWinPosition;
    };

    nav.onResize = function () {
        this.winResize = false;
        this.winHeight = this.win.height();
        this.docHeight = $(document).height();
    };

    nav.hashChange = function () {
        this.linkScroll = true;
        this.win.one('hashchange', function () {
            this.linkScroll = false;
        });
    };

    nav.toggleCurrent = function (elem) {
        var parent_li = elem.closest('li');
        parent_li.siblings('li.current').removeClass('current');
        parent_li.siblings().find('li.current').removeClass('current');
        parent_li.find('> ul li.current').removeClass('current');
        parent_li.toggleClass('current');
    }

    return nav;
};

module.exports.ThemeNav = ThemeNav();

if (typeof(window) != 'undefined') {
    window.SphinxRtdTheme = {
        Navigation: module.exports.ThemeNav,
        // TODO remove this once static assets are split up between the theme
        // and Read the Docs. For now, this patches 0.3.0 to be backwards
        // compatible with a pre-0.3.0 layout.html
        StickyNav: module.exports.ThemeNav,
    };
}


// requestAnimationFrame polyfill by Erik Möller. fixes from Paul Irish and Tino Zijdel
// https://gist.github.com/paulirish/1579671
// MIT license

(function() {
    var lastTime = 0;
    var vendors = ['ms', 'moz', 'webkit', 'o'];
    for(var x = 0; x < vendors.length && !window.requestAnimationFrame; ++x) {
        window.requestAnimationFrame = window[vendors[x]+'RequestAnimationFrame'];
        window.cancelAnimationFrame = window[vendors[x]+'CancelAnimationFrame']
                                   || window[vendors[x]+'CancelRequestAnimationFrame'];
    }

    if (!window.requestAnimationFrame)
        window.requestAnimationFrame = function(callback, element) {
            var currTime = new Date().getTime();
            var timeToCall = Math.max(0, 16 - (currTime - lastTime));
            var id = window.setTimeout(function() { callback(currTime + timeToCall); },
              timeToCall);
            lastTime = currTime + timeToCall;
            return id;
        };

    if (!window.cancelAnimationFrame)
        window.cancelAnimationFrame = function(id) {
            clearTimeout(id);
        };
}());

$(".sphx-glr-thumbcontainer").removeAttr("tooltip");
$("table").removeAttr("border");

},{"jquery":"jquery"}]},{},[1,2,3,4,5,6,7,8,"pytorch-sphinx-theme"]);
