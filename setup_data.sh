#!/bin/bash
# Download and setup Spider dataset

set -e

echo "========================================="
echo "Setting up Spider Dataset"
echo "========================================="

# Create directories
mkdir -p data
mkdir -p databases

# Download Spider dataset from official source
echo "Downloading Spider dataset from official source..."
echo "URL: https://drive.google.com/file/d/1oi9J1jZP9TyM35L85CL3qeGWl2jqlnL6"
cd data

# Use gdown for Google Drive downloads (more reliable)
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown for Google Drive downloads..."
    pip install gdown
fi

gdown "1oi9J1jZP9TyM35L85CL3qeGWl2jqlnL6" -O spider.zip

# Extract
echo "Extracting Spider dataset..."
unzip -o -q spider.zip

# Move database folder to correct location
echo "Organizing files..."
if [ -d "database" ]; then
    mv database/* ../databases/ 2>/dev/null || true
    rm -rf database
fi

# Cleanup
rm -f spider.zip

cd ..

# Verify required files exist
echo "Verifying dataset files..."
if [ -f "data/dev.json" ] && [ -f "data/train_spider.json" ] && [ -f "data/tables.json" ]; then
    echo "✓ Dataset files found"
else
    echo "✗ Missing required dataset files"
    exit 1
fi

echo ""
echo "✅ Spider dataset setup complete!"
echo ""
echo "Files created:"
echo "  - data/train_spider.json"
echo "  - data/dev.json"
echo "  - data/tables.json"
echo "  - databases/* (multiple SQLite databases)"
