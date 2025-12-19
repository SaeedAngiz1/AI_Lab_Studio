# Deployment Guide

This guide explains how to deploy AI Lab Studio to various platforms.

## Streamlit Cloud Deployment

### Prerequisites
- A GitHub account
- Your repository pushed to GitHub
- A Streamlit Cloud account (free tier available)

### Steps

1. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**
   - Sign in with your GitHub account
   - Click "New app"

2. **Configure your app**
   - **Repository**: Select `SaeedAngiz1/AI_Lab_Studio`
   - **Branch**: `main` or `master`
   - **Main file**: `app.py`
   - **Python version**: `3.10` (or latest supported)

3. **Deploy**
   - Click "Deploy"
   - Wait for the build to complete
   - Your app will be live at `https://your-app-name.streamlit.app`

### Environment Variables
If you need to set environment variables:
- Go to your app settings in Streamlit Cloud
- Add secrets in the "Secrets" section
- Access them in your app using `st.secrets`

## Docker Deployment

### Build Docker Image

```bash
docker build -t ai-lab-studio .
```

### Run Container

```bash
docker run -p 8501:8501 ai-lab-studio
```

## Local Deployment

### Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Production

```bash
# Using gunicorn (if using custom server)
gunicorn app:app

# Or using Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## GitHub Pages (Static Site)

Note: Streamlit apps require a server, so GitHub Pages won't work directly. Consider:
- Using Streamlit Cloud (recommended)
- Converting to a static site generator
- Using a different hosting platform

## Heroku Deployment

1. **Create a Procfile**:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Deploy**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

## Railway Deployment

1. **Connect your GitHub repository** to Railway
2. **Set build command**: `pip install -r requirements.txt`
3. **Set start command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. **Deploy**

## Vercel Deployment

Vercel doesn't natively support Streamlit. Consider:
- Using Streamlit Cloud (recommended)
- Converting to a different framework
- Using a custom server setup

## AWS/GCP/Azure Deployment

### AWS EC2
1. Launch an EC2 instance
2. Install Python and dependencies
3. Run Streamlit with a reverse proxy (nginx)
4. Configure security groups

### Google Cloud Run
1. Create a Dockerfile
2. Build and push container
3. Deploy to Cloud Run
4. Set port to 8501

### Azure App Service
1. Create a Web App
2. Configure Python runtime
3. Deploy via GitHub Actions or Azure CLI

## Continuous Deployment

The repository includes GitHub Actions workflows for:
- **CI**: Automated testing and linting
- **Deploy**: Automated deployment to Streamlit Cloud
- **Release**: Automated package releases

Configure secrets in GitHub repository settings for automated deployments.

## Troubleshooting

### Common Issues

1. **Port already in use**
   - Change port: `streamlit run app.py --server.port 8502`

2. **Dependencies not found**
   - Ensure `requirements.txt` is up to date
   - Run `pip install -r requirements.txt`

3. **Import errors**
   - Check Python path
   - Verify all dependencies are installed

4. **Memory issues**
   - Increase memory limits in deployment platform
   - Optimize data loading

## Monitoring

- Use Streamlit Cloud's built-in analytics
- Set up error tracking (Sentry, etc.)
- Monitor resource usage
- Set up alerts for downtime

