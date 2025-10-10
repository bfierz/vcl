#!/usr/bin/env bash
set -euo pipefail

# Environment
export DISPLAY=${DISPLAY:-:1}
export VNC_PASSWORD=${VNC_PASSWORD:-localvnc}
export NOVNC_DIR=${NOVNC_DIR:-/usr/share/novnc}
export WEBSOCKIFY_CMD=${WEBSOCKIFY_CMD:-/usr/bin/websockify}

# Create a simple openbox session if not present
mkdir -p ~/.config/openbox
if [ ! -f ~/.config/openbox/autostart ]; then
  cat > ~/.config/openbox/autostart <<'EOF'
# Launch a minimal desktop environment
xsetroot -solid "#202020" &
# handy terminal
xterm -fa Monospace -fs 11 -geometry 100x30+50+50 &
EOF
fi

# Set VNC password for x11vnc
mkdir -p ~/.vnc
x11vnc -storepasswd "${VNC_PASSWORD}" ~/.vnc/passwd

# Start everything under supervisord (xvfb, openbox, x11vnc, novnc proxy)
echo "Starting desktop via noVNC on port 6080..."
sudo /usr/bin/supervisord -c /etc/supervisor/conf.d/novnc.conf

# Print helpful message
cat <<EOT

noVNC is now running.

1) In Codespaces, check the Ports tab. Port 6080 should auto-open in your browser.
2) Inside the browser desktop, open a terminal (xterm) and run your app, e.g.:

   ./build/my_app

Tip: If your app needs a larger display, edit DISPLAY=:1 screen size in supervisord.conf

EOT
