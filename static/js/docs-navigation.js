/**
 * Documentation navigation utilities for SpectraML
 */

document.addEventListener('DOMContentLoaded', function() {
    // Add ID attributes to headings if they don't have them already
    const content = document.querySelector('.markdown-content');
    if (content) {
        const headings = content.querySelectorAll('h1, h2, h3, h4, h5, h6');
        
        headings.forEach(heading => {
            if (!heading.id) {
                const id = heading.textContent
                    .toLowerCase()
                    .replace(/[^\w\s-]/g, '')
                    .replace(/\s+/g, '-');
                heading.id = id;
            }
        });
    }
    
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 100, // Offset for the navbar
                    behavior: 'smooth'
                });
                
                // Update URL hash without jumping
                history.pushState(null, null, targetId);
            }
        });
    });
    
    // Generate table of contents if TOC placeholder exists
    const tocPlaceholder = document.getElementById('toc-placeholder');
    if (tocPlaceholder && content) {
        const headings = content.querySelectorAll('h2, h3');
        if (headings.length > 0) {
            // Create TOC element
            const toc = document.createElement('div');
            toc.className = 'toc mb-6';
            
            // Create TOC title
            const tocTitle = document.createElement('div');
            tocTitle.className = 'toc-title';
            tocTitle.textContent = 'On This Page';
            toc.appendChild(tocTitle);
            
            // Create TOC list
            const tocList = document.createElement('ul');
            toc.appendChild(tocList);
            
            // Add TOC items
            let currentLevel = 2;
            let currentList = tocList;
            let parentList = null;
            
            headings.forEach(heading => {
                const level = parseInt(heading.tagName.substring(1));
                
                // Create new list item
                const listItem = document.createElement('li');
                listItem.className = `toc-level-${level}`;
                
                // Create link to heading
                const link = document.createElement('a');
                link.href = `#${heading.id}`;
                link.textContent = heading.textContent;
                listItem.appendChild(link);
                
                // Handle different heading levels
                if (level > currentLevel) {
                    // Create sub-list
                    parentList = currentList;
                    const newList = document.createElement('ul');
                    parentList.lastChild.appendChild(newList);
                    currentList = newList;
                } else if (level < currentLevel) {
                    // Go back to parent list
                    currentList = parentList || tocList;
                }
                
                currentLevel = level;
                currentList.appendChild(listItem);
            });
            
            // Replace placeholder with TOC
            tocPlaceholder.parentNode.replaceChild(toc, tocPlaceholder);
        } else {
            // No headings, remove placeholder
            tocPlaceholder.remove();
        }
    }
    
    // Highlight current section in navigation menu
    const pathname = window.location.pathname;
    const navLinks = document.querySelectorAll('.guide-nav-item a');
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === pathname) {
            link.parentElement.classList.add('active');
        }
    });
});
