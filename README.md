# bot-code

The code for controlling the robots.

To use, clone the robus-core repository and place it in this repository.
It functions as the core connecting all the nodes in this repository.

```bash
cd bot-code
git clone https://github.com/Robocup-Junior-Open-League/robus-core
```

The core can be updated independetly from the nodes by navigating to the robus-core directory and pulling the latest changes.

```bash
cd bot-code/robus-core
git pull
```

Python libraries need to be installed to run:

```bash
cd bot-code
pip install -r requirements.txt
```

To start the bot, run the following commands:

Linux:

```bash
cd bot-code
source ./robus-core/setup/start.sh
```

Windows:

```bash
cd bot-code
robus-core/setup/start.bat
```
