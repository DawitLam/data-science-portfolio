# 🚀 GitHub Setup Commands

# Step 1: Remove the old remote
git remote remove origin

# Step 2: Add your new repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/ml-healthcare-portfolio.git

# Step 3: Push to GitHub
git branch -M main
git push -u origin main

# Alternative: If you prefer SSH (requires SSH key setup)
# git remote add origin git@github.com:YOUR_USERNAME/ml-healthcare-portfolio.git
