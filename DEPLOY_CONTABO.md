# Deploy to Contabo VPS

This project is designed to run on a VPS through Git and Docker Compose.

## 1. Push from local PC to GitHub

Create an empty GitHub repository, then run from this project directory:

```powershell
git remote add origin https://github.com/YOUR_NAME/market-intel.git
git branch -M main
git push -u origin main
```

If GitHub asks for credentials, use GitHub Desktop login, Git Credential Manager, or a personal access token.

## 2. Connect to Contabo

From Windows PowerShell:

```powershell
ssh root@SERVER_IP
```

For a non-root user:

```powershell
ssh username@SERVER_IP
```

## 3. Prepare Ubuntu 24.04

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install docker.io docker-compose-plugin git ufw -y
sudo systemctl enable docker
sudo systemctl start docker
```

Allow SSH and the dashboard port:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 8000/tcp
sudo ufw enable
sudo ufw status
```

## 4. Clone and configure

```bash
git clone https://github.com/YOUR_NAME/market-intel.git
cd market-intel
cp .env.example .env
```

Check `.env` before starting. The default paper-lab settings are safe: no live trading, local paper execution, collector interval 5 seconds.

Set a private dashboard password:

```bash
nano .env
```

Use a strong value:

```text
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=PUT_A_LONG_PRIVATE_PASSWORD_HERE
DASHBOARD_SESSION_SECONDS=86400
DASHBOARD_COOKIE_SECURE=false
```

The dashboard uses a browser session cookie. It is not written as a persistent cookie, so after closing the browser session the next dashboard visit asks for login again. Use the `Вийти` link to end the session immediately.

Set `DASHBOARD_COOKIE_SECURE=true` only after the dashboard is behind HTTPS. Keep it `false` for plain `http://SERVER_IP:8000`, otherwise the browser will not send the cookie.

Optional: restrict the dashboard to specific public IP addresses:

```text
DASHBOARD_ALLOWED_IPS=203.0.113.10,198.51.100.25
```

Leave `DASHBOARD_ALLOWED_IPS` empty if your home or mobile IP addresses change often. The password protection still applies to the dashboard and all `/api/*` endpoints.

## 5. Start services

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f
```

Expected services:

```text
postgres
redis
collector
research
dashboard
```

Open the dashboard:

```text
http://SERVER_IP:8000
```

## 6. Update deployment later

On the local PC:

```powershell
git add .
git commit -m "Update"
git push
```

On the VPS:

```bash
cd market-intel
git pull
docker compose up -d --build
docker compose ps
```

## 7. Useful operations

View dashboard logs:

```bash
docker compose logs -f dashboard
```

View collector logs:

```bash
docker compose logs -f collector
```

View Telegram notifier logs:

```bash
docker compose logs -f telegram-notifier
```

Restart all services:

```bash
docker compose restart
```

Stop all services:

```bash
docker compose down
```

## 8. Data to keep backed up

The important runtime state is:

```text
data/
models/
configs/
postgres_data Docker volume
redis_data Docker volume
```

Quick file backup:

```bash
tar -czf market-intel-files-backup.tar.gz data models configs
```

Docker volume backup can be added after the first production run, once the exact server path and backup destination are chosen.

## 9. Telegram Signal Notifications

Create a bot in Telegram:

1. Open `@BotFather`.
2. Run `/newbot`.
3. Copy the bot token.
4. Send any message to your new bot from your Telegram account.
5. Open this URL in a browser, replacing the token:

```text
https://api.telegram.org/botBOT_TOKEN/getUpdates
```

Find your numeric `chat.id`, then update `.env` on the server:

```text
TELEGRAM_NOTIFICATIONS_ENABLED=true
TELEGRAM_BOT_TOKEN=BOT_TOKEN
TELEGRAM_CHAT_ID=YOUR_CHAT_ID
TELEGRAM_MIN_CONFIDENCE=0.60
TELEGRAM_NOTIFIER_INTERVAL_SECONDS=5
```

Restart the notifier:

```bash
docker compose up -d --build telegram-notifier
docker compose logs -f telegram-notifier
```

The notifier sends one message per opened paper position when `signal_execution_audit.csv` contains `executed=True`, `reason=OPENED`, and confidence is at least `TELEGRAM_MIN_CONFIDENCE`. Sent notifications are tracked in:

```text
data/reports/telegram_notifications.csv
```
