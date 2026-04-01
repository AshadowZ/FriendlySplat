#!/bin/bash
# entrypoint.sh

# Ensure the target directory for critical compiled artifacts exists
mkdir -p /app/FriendlySplat/gsplat

# Copy the prebuilt csrc.so from the image artifacts to the mounted directory
cp /opt/artifacts/gsplat/csrc.so /app/FriendlySplat/gsplat/csrc.so

# Execute the container's startup command (bash or other)
exec "$@"