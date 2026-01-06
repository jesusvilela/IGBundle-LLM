# Deployment Guide

This repository is configured to be deployed via **GitHub Pages** using the `/docs` folder source.

## How to Enable

1. Go to your repository **Settings** tab.
2. Click on **Pages** in the left sidebar (under "Code and automation").
3. Under **Build and deployment**:
    - **Source**: Select `Deploy from a branch`.
    - **Branch**: Select `main` (or your current branch).
    - **Folder**: Select `/docs`.
4. Click **Save**.

## Why this structure?
The `/docs` folder contains a pre-built static site (HTML/CSS/JS) that acts as:
- **Project Wiki**: Full HTML version of the thesis (`wiki.html`).
- **Launcher Portal**: A guide to launching the local optimized engine (`demo.html`).
- **Landing Page**: Project overview (`index.html`).

## Local Development
To preview the site locally, simply open `docs/index.html` in your web browser.
