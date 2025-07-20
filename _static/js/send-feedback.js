// A function to open new issue in GitHub based on {{feedback_url}}.
// Activated when you click the "Send Feedback" button in the footer.
function openGitHubIssue() {
    var baseUrl = document.querySelector('.pytorch-body').getAttribute('data-feedback-url');
    if (!baseUrl) {
        console.error('Feedback URL not found');
        return;
    }
    var pageUrl = window.location.href;
    var pageTitle = document.querySelector('h1')?.textContent.split('#')[0].trim() || 'Page';
    var issueTitle = encodeURIComponent(`Feedback about ${pageTitle}`);
    var issueBody = encodeURIComponent(`There is the following issue on this page: ${pageUrl}`);
    var labels = encodeURIComponent("module: docs,page-feedback");
    var feedbackUrl = `${baseUrl}/issues/new?title=${issueTitle}&body=${issueBody}&labels=${labels}`;

    // Track event in Google Analytics
    window.dataLayer = window.dataLayer || [];
    window.dataLayer.push({
        'event': 'send_feedback',
        'page_title': pageTitle,
        'page_location': pageUrl
    });
    console.log('Data Layer event pushed: send_feedback', pageUrl);

    window.open(feedbackUrl, '_blank');
}
