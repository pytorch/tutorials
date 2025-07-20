$(document).ready(function() {
  // Build an array from each tag that's present
  var tagList = [];
  var tutorialsPerPage = 8;
  var currentPage = 1;

  // Check if tutorial containers exist
  if ($(".tutorials-card-container").length > 0) {
    tagList = $(".tutorials-card-container").map(function() {
      if ($(this).data("tags")) {
        return $(this).data("tags").split(",").map(function(item) {
          return item.trim();
        });
      }
      return [];
    }).get();
  } else {
    console.log("No tutorial card containers found");
  }

  // Flatten the array of arrays
  tagList = [].concat.apply([], tagList);

  function unique(value, index, self) {
    return self.indexOf(value) == index && value != "";
  }

  // Only return unique tags
  var tags = tagList.sort().filter(unique);

  console.log("Found tags:", tags);

  // Add filter buttons to the top of the page for each tag
  function createTagMenu() {
    // Check if "All" button already exists
    if ($(".tutorial-filter-menu .filter-btn[data-tag='all']").length === 0) {
      // Add "All" button first
      $(".tutorial-filter-menu").append("<div class='tutorial-filter filter-btn filter selected' data-tag='all'>All</div>");
    }

    if (tags.length > 0) {
      tags.forEach(function(item){
        // Skip adding "all" tag if it's in the tags list
        if (item.toLowerCase() !== "all") {
          $(".tutorial-filter-menu").append(" <div class='tutorial-filter filter-btn filter' data-tag='" + item + "'>" + item + "</div>");
        }
      });
    } else {
      console.log("No tags found to create menu");
    }
  }

  createTagMenu();

  // Create pagination controls
  function createPagination(totalItems) {
    var totalPages = Math.ceil(totalItems / tutorialsPerPage);

    // Clear existing pagination
    $(".tutorials-pagination").remove();

    if (totalPages <= 1) return; // Don't show pagination if only one page

    var paginationHtml = '<div class="tutorials-pagination">';
    paginationHtml += '<button class="page-btn prev" disabled>&laquo; Prev</button>';

    for (var i = 1; i <= totalPages; i++) {
      var activeClass = i === currentPage ? 'active' : '';
      paginationHtml += '<button class="page-btn page-number ' + activeClass + '" data-page="' + i + '">' + i + '</button>';
    }

    paginationHtml += '<button class="page-btn next">Next &raquo;</button>';
    paginationHtml += '</div>';

    // Append pagination after tutorials container
    $(".tutorials-card-container").last().after(paginationHtml);

    // Add event listeners
    $(".page-btn.page-number").on("click", function() {
      currentPage = parseInt($(this).data("page"));
      showCurrentPage();
    });

    $(".page-btn.prev").on("click", function() {
      if (currentPage > 1) {
        currentPage--;
        showCurrentPage();
      }
    });

    $(".page-btn.next").on("click", function() {
      if (currentPage < totalPages) {
        currentPage++;
        showCurrentPage();
      }
    });
  }

  // Function to show current page of tutorials
  function showCurrentPage() {
    var visibleTutorials = $(".tutorials-card-container").filter(function() {
      // Check if this tutorial should be visible based on current filter
      var selectedTags = $(".filter-btn.selected").map(function() {
        return $(this).data("tag");
      }).get();

      if (selectedTags.includes("all") || selectedTags.length === 0) {
        return true;
      } else {
        var cardTags = $(this).data("tags").split(",").map(function(tag) {
          return tag.trim();
        });

        return cardTags.some(function(tag) {
          return selectedTags.includes(tag);
        });
      }
    });

    // Hide all tutorials first
    $(".tutorials-card-container").hide();

    // Show only the ones for current page
    var startIndex = (currentPage - 1) * tutorialsPerPage;
    var endIndex = startIndex + tutorialsPerPage;
    visibleTutorials.slice(startIndex, endIndex).show();

    // Update pagination buttons
    $(".page-btn.page-number").removeClass("active");
    $(".page-btn.page-number[data-page='" + currentPage + "']").addClass("active");

    // Enable/disable prev/next buttons
    $(".page-btn.prev").prop("disabled", currentPage === 1);
    $(".page-btn.next").prop("disabled", currentPage === Math.ceil(visibleTutorials.length / tutorialsPerPage));
  }


  // Show tutorials and initialize pagination
  function filterAndShowTutorials() {
    var selectedTags = $(".filter-btn.selected").map(function() {
      return $(this).data("tag");
    }).get();

    // Reset to first page when filtering
    currentPage = 1;

    if (selectedTags.includes("all") || selectedTags.length === 0) {
      // Show all tutorials
      $(".tutorials-card-container").show();
    } else {
      // Filter by selected tags
      $(".tutorials-card-container").each(function() {
        var cardTags = $(this).data("tags").split(",").map(function(tag) {
          return tag.trim();
        });

        var hasSelectedTag = cardTags.some(function(tag) {
          return selectedTags.includes(tag);
        });

        $(this).toggle(hasSelectedTag);
      });
    }

    // Create pagination based on visible tutorials
    createPagination($(".tutorials-card-container:visible").length);
    showCurrentPage();
  }

  // Initial display
  filterAndShowTutorials();

  // Add click handler for filter buttons
  $(".filter-btn").on("click", function() {
    var selectedTag = $(this).data("tag");

    // If "All" button is clicked, clear all selections and select "All"
    if (selectedTag === "all") {
      $(".filter-btn").removeClass("selected");
      $(this).addClass("selected");
    } else {
      // Remove "All" selection when other tags are clicked
      $(".filter-btn[data-tag='all']").removeClass("selected");
      // Toggle selected class
      $(this).toggleClass("selected");
    }

    filterAndShowTutorials();
  });

  // Remove hyphens if they are present in the filter buttons
  $(".tags").each(function(){
    var tags = $(this).text().split(",");
    tags.forEach(function(tag, i) {
      tags[i] = tags[i].replace(/-/, ' ');
    });
    $(this).html(tags.join(", "));
  });

  // Remove hyphens if they are present in the card body
  $(".tutorial-filter").each(function(){
    var tag = $(this).text();
    $(this).html(tag.replace(/-/, ' '));
  });
});
