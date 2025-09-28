# GitHub Pages Setup Instructions

To enable the ROI Calculator at `https://arbitrium-framework.github.io/arbitrium/calculator.html`:

## Deployment Method

This project uses **MkDocs** to build documentation. MkDocs builds to the `/site` directory (not `/docs`).

## Setup via GitHub Actions (Recommended)

1. **Enable GitHub Pages**
   - Navigate to https://github.com/arbitrium-framework/arbitrium
   - Click "Settings" tab
   - In left sidebar, click "Pages"
   - Under "Source", select "GitHub Actions"

2. **Set up MkDocs GitHub Action**
   - Create `.github/workflows/docs.yml` with MkDocs deployment workflow
   - The workflow should run `mkdocs build` which creates the `/site` directory
   - GitHub Actions will automatically deploy the site

3. **Wait for Deployment**
   - Push changes to `main` branch
   - GitHub Actions will build and deploy (takes 1-2 minutes)
   - Visit: https://arbitrium-framework.github.io/arbitrium/

4. **Test Calculator**
   - Navigate to https://arbitrium-framework.github.io/arbitrium/calculator.html
   - Verify calculations work correctly

## Manual Deployment (Alternative)

If deploying manually:

```bash
# Build the site
mkdocs build

# Deploy to gh-pages branch
mkdocs gh-deploy
```

## Updating the Site

After GitHub Pages is enabled, any changes to files in the `docs/` directory will require rebuilding:
- **With GitHub Actions**: Push to `main` (builds automatically)
- **Manual**: Run `mkdocs gh-deploy`

Note: MkDocs builds from the `docs/` source directory to the `site/` output directory.

## Custom Domain (Optional)

If you want to use a custom domain:

1. Add custom domain in GitHub Pages settings
2. MkDocs will automatically create the `CNAME` file during deployment
3. Configure DNS records with your domain registrar
4. Enable HTTPS in GitHub Pages settings

See: https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site

## Verification

Once enabled, the calculator URL should be linked in README.md:

- Main link: https://arbitrium-framework.github.io/arbitrium/calculator.html
- Local preview: Run `mkdocs serve` and visit http://127.0.0.1:8000/calculator.html
