document.addEventListener('DOMContentLoaded', () => {
    console.log('ManifoldGL Documentation Loaded');

    // Smooth scroll
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Demo status checker
    const demoStatus = document.getElementById('demo-status');
    if (demoStatus) {
        // Simple check if localhost:7860 is reachable
        const checkServer = async () => {
            try {
                // We can't actually fetch localhost from a static site due to CORS mostly,
                // but the iframe will show the error natively.
                // We'll just show a helper message.
                demoStatus.innerHTML = `
                    ✅ <strong>Active Mode</strong>: Connecting to <code>localhost:7860</code>. 
                    If the demo below doesn't load, please run <code>launch_optimized.bat</code>.
                `;
            } catch (e) {
                demoStatus.classList.add('error');
                demoStatus.innerHTML = `⚠️ Backend unreachable. Run <code>launch_optimized.bat</code> to start the model.`;
            }
        };
        checkServer();
    }
});
