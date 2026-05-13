#!/bin/bash
# ===============================
# FIX PERMISSIONS SCRIPT
# ===============================
# Corrige problemas de permisos en carpetas creadas por Docker
# Uso: ./fix_permission.sh

set -e

echo "=================================="
echo "🔒 FIXING PERMISSIONS"
echo "=================================="

# Detect user and group
USER_ID=$(id -u)
GROUP_ID=$(id -g)

echo "👤 Setting ownership to $USER_ID:$GROUP_ID"

# Fix ownership for critical directories and files
# Using sudo because files might be owned by root
sudo chown -R $USER_ID:$GROUP_ID exports logs runs dataset_master.sqlite 2>/dev/null || true

# Fix permissions
echo "🔑 Setting read/write permissions"
sudo chmod -R u+rwX exports logs runs 2>/dev/null || true

echo ""
echo "✅ Permissions fixed!"
echo "   You should now be able to edit/delete files in exports/, logs/ and runs/"
echo "=================================="