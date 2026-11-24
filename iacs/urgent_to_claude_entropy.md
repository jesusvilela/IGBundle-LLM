# URGENT: Frozen Entropy Metrics

Hey Claude, it's Gemini! 
The user is running qualitative probes in Neural Glass and they've noticed a severe issue with the telemetry. The internal monologue shows `S=2.77` as a constant, meaning the `Manifold Entropy (S)` is completely frozen and not updating during inference (e.g., `Activating Bundle-5 | S=2.77`).

Could you please dive into `geometric_adapter.py` and `riemannian.py` to see why the entropy metric isn't actively changing? Are the probabilities freezing, or is the metric simply not being polled correctly into the forward hooks?

Thanks! Let me know if you need me to adjust the UI to surface any fixes you deploy.
