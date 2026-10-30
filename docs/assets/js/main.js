document.addEventListener('DOMContentLoaded', () => {
    console.log('IGBundle Documentation Loaded');

    // Smooth scroll
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.getElementById(this.getAttribute('href').slice(1));
            if (target) target.scrollIntoView({ behavior: 'smooth' });
        });
    });

});
