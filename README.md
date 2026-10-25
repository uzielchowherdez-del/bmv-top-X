
# Bot BMV Automático - Lunes a Viernes 8:35 AM

## Cómo instalarlo en 5 minutos:

1. Crea un repositorio nuevo en GitHub (ej: bmv-top-bot) y sube estos 3 archivos: bot.py, requirements.txt y la carpeta .github/workflows/daily.yml

2. En tu repositorio ve a Settings -> Secrets and variables -> Actions -> New repository secret
   - Crea TELEGRAM_TOKEN con tu token de @BotFather
   - Crea CHAT_ID con tu ID (el que sacaste con @userinfobot)

3. Ve a Actions -> TOP BMV Diario -> Run workflow -> Run para probar. Te debe llegar el mensaje de hoy a Telegram.

4. Listo. Ya corre solo de lunes a viernes a las 8:35 AM hora México sin abrir Colab.

## Si falla Yahoo Finance:
GitHub a veces bloquea Yahoo. Si no te llega mensaje, revisa en Actions el log. Alternativa: usar PythonAnywhere con la misma tarea programada.
