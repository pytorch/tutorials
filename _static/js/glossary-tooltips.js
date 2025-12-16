/**
 * Glossary Tooltips - Custom tooltip implementation for Sphinx glossary terms
 *
 * This script adds hover tooltips to glossary term references throughout the documentation.
 * When hovering over a glossary term link, it fetches the definition from the glossary page
 * and displays it in a tooltip without requiring navigation.
 */

(function () {
    'use strict';

    // Cache for glossary definitions to avoid repeated fetches
    const glossaryCache = {};
    let glossaryContent = null;
    let tooltip = null;
    let currentTarget = null;
    let hideTimeout = null;

    /**
     * Create the tooltip element
     */
    function createTooltip() {
        tooltip = document.createElement('div');
        tooltip.className = 'glossary-tooltip';
        tooltip.style.cssText = `
            position: absolute;
            display: none;
            background: #2c2c2c;
            color: #e8e8e8;
            padding: 12px 16px;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            max-width: 400px;
            z-index: 10000;
            font-size: 14px;
            line-height: 1.5;
            pointer-events: none;
        `;
        document.body.appendChild(tooltip);
    }

    /**
     * Fetch and cache glossary content using iframe (works with file:// URLs)
     */
    async function fetchGlossaryContent() {
        if (glossaryContent) {
            return glossaryContent;
        }

        return new Promise((resolve, reject) => {
            try {
                // Create hidden iframe to load glossary
                const iframe = document.createElement('iframe');
                iframe.style.display = 'none';

                // Determine glossary URL - find the base URL by looking for common patterns
                const currentPath = window.location.pathname;
                let basePath = '';

                // Find the root of the documentation
                const pathParts = currentPath.split('/');
                for (let i = 0; i < pathParts.length; i++) {
                    if (pathParts[i] === '_build') {
                        // For local builds, glossary is at _build/html/glossary.html
                        basePath = pathParts.slice(0, i + 2).join('/') + '/';
                        break;
                    }
                }

                // If we couldn't find _build, try to find common doc directories
                if (!basePath) {
                    const knownDirs = ['advanced', 'beginner', 'intermediate', 'recipes', 'prototype', 'unstable'];
                    for (let i = pathParts.length - 1; i >= 0; i--) {
                        if (knownDirs.includes(pathParts[i])) {
                            basePath = pathParts.slice(0, i).join('/') + '/';
                            break;
                        }
                    }
                }

                // Fallback to going up directories based on current location
                if (!basePath) {
                    basePath = currentPath.substring(0, currentPath.lastIndexOf('/') + 1) + '../';
                }

                const glossaryUrl = window.location.origin + basePath + 'glossary.html';

                console.log('Loading glossary from:', glossaryUrl);

                iframe.onload = function () {
                    try {
                        const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;

                        // Verify we got valid content
                        if (!iframeDoc.getElementById('term-ATen') && !iframeDoc.getElementById('term-JIT')) {
                            console.warn('Glossary loaded but no terms found');
                        } else {
                            console.log('Glossary content successfully loaded');
                        }

                        // Clone the body content before removing iframe
                        const clonedBody = iframeDoc.body.cloneNode(true);

                        // Create a container to hold the content
                        const container = document.createElement('div');
                        container.innerHTML = clonedBody.innerHTML;
                        container.style.display = 'none';
                        container.id = 'glossary-content-cache';
                        document.body.appendChild(container);

                        glossaryContent = container;

                        // Remove iframe after cloning
                        if (iframe.parentNode) {
                            iframe.parentNode.removeChild(iframe);
                        }

                        resolve(glossaryContent);
                    } catch (error) {
                        console.error('Error accessing iframe content:', error);
                        reject(error);
                    }
                };

                iframe.onerror = function (error) {
                    console.error('Error loading glossary iframe:', error);
                    reject(error);
                };

                document.body.appendChild(iframe);
                iframe.src = glossaryUrl;

                // Timeout after 5 seconds
                setTimeout(() => {
                    if (!glossaryContent) {
                        console.error('Glossary loading timeout');
                        if (iframe.parentNode) {
                            iframe.parentNode.removeChild(iframe);
                        }
                        reject(new Error('Timeout loading glossary'));
                    }
                }, 5000);

            } catch (error) {
                console.error('Failed to create glossary iframe:', error);
                reject(error);
            }
        });
    }

    /**
     * Extract definition text from glossary entry
     */
    function getDefinitionText(termId, container) {
        if (glossaryCache[termId]) {
            return glossaryCache[termId];
        }

        try {
            // Find the term definition in the glossary
            // container is a div element, so use querySelector instead of getElementById
            const termElement = container.querySelector('#' + CSS.escape(termId));
            if (!termElement) {
                console.warn('Term not found:', termId);
                return null;
            }

            // Get the definition - it's in the <dd> that follows the <dt>
            let definitionElement = termElement.nextElementSibling;
            while (definitionElement && definitionElement.tagName !== 'DD') {
                definitionElement = definitionElement.nextElementSibling;
            }

            if (!definitionElement) {
                // Try looking for the parent dt and its sibling dd
                const parentDt = termElement.closest('dt');
                if (parentDt) {
                    definitionElement = parentDt.nextElementSibling;
                    while (definitionElement && definitionElement.tagName !== 'DD') {
                        definitionElement = definitionElement.nextElementSibling;
                    }
                }
            }

            if (!definitionElement) {
                console.warn('Definition element not found for:', termId);
                return null;
            }

            // Clone the element to manipulate it without affecting the original
            const clone = definitionElement.cloneNode(true);

            // Remove any internal reference links (keep the text but remove the link)
            clone.querySelectorAll('a.reference.internal').forEach(link => {
                const text = document.createTextNode(link.textContent);
                link.parentNode.replaceChild(text, link);
            });

            // Get clean text with basic formatting
            let text = clone.textContent.trim();

            // Limit length and add ellipsis if needed
            const maxLength = 300;
            if (text.length > maxLength) {
                text = text.substring(0, maxLength).trim() + '...';
            }

            glossaryCache[termId] = text;
            return text;
        } catch (error) {
            console.error('Error extracting definition:', error);
            return null;
        }
    }

    /**
     * Show tooltip at the given position
     */
    function showTooltip(text, target) {
        if (!tooltip || !text) {
            return;
        }

        clearTimeout(hideTimeout);

        tooltip.textContent = text;
        tooltip.style.display = 'block';

        // Position tooltip
        const rect = target.getBoundingClientRect();
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;

        let top = rect.bottom + scrollTop + 8;
        let left = rect.left + scrollLeft + (rect.width / 2);

        // Adjust position if tooltip would go off-screen
        const tooltipRect = tooltip.getBoundingClientRect();

        // Horizontal adjustment
        if (left + tooltipRect.width / 2 > window.innerWidth) {
            left = window.innerWidth - tooltipRect.width - 10 + scrollLeft;
        } else if (left - tooltipRect.width / 2 < 0) {
            left = 10 + scrollLeft;
        } else {
            left = left - tooltipRect.width / 2;
        }

        // Vertical adjustment - show above if no room below
        if (rect.bottom + tooltipRect.height + 16 > window.innerHeight + scrollTop) {
            top = rect.top + scrollTop - tooltipRect.height - 8;
        }

        tooltip.style.top = top + 'px';
        tooltip.style.left = left + 'px';

        currentTarget = target;
    }

    /**
     * Hide tooltip with delay
     */
    function hideTooltip() {
        hideTimeout = setTimeout(() => {
            if (tooltip) {
                tooltip.style.display = 'none';
                currentTarget = null;
            }
        }, 100);
    }

    /**
     * Handle mouse enter on glossary term link
     */
    async function handleMouseEnter(event) {
        const link = event.currentTarget;
        const href = link.getAttribute('href');

        // Check if this is a glossary term link
        if (!href || !href.includes('glossary.html#term-')) {
            return;
        }

        // Extract term ID from href
        const termId = href.split('#')[1];
        if (!termId) {
            return;
        }

        // Show loading indicator for slow networks
        const loadingText = 'Loading definition...';
        showTooltip(loadingText, link);

        // Fetch glossary content if not already cached
        const doc = await fetchGlossaryContent();
        if (!doc) {
            hideTooltip();
            return;
        }

        // Get definition text
        const definition = getDefinitionText(termId, doc);
        if (definition && currentTarget === link) {
            showTooltip(definition, link);
        } else {
            hideTooltip();
        }
    }

    /**
     * Initialize tooltips for all glossary term links
     */
    function initializeGlossaryTooltips() {
        // Create tooltip element
        createTooltip();

        // Find all glossary term links
        const glossaryLinks = document.querySelectorAll('a.reference.internal[href*="glossary.html#term-"]');

        glossaryLinks.forEach(link => {
            link.addEventListener('mouseenter', handleMouseEnter);
            link.addEventListener('mouseleave', hideTooltip);
        });

        console.log(`Initialized glossary tooltips for ${glossaryLinks.length} terms`);
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeGlossaryTooltips);
    } else {
        initializeGlossaryTooltips();
    }

})();
