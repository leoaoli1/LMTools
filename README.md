## LMTools

LMTools is a repository designed to use large models as practical tools.

## Setup

```bash
git clone https://github.com/leoaoli1/LMTools.git
pip install -r requirements.txt
```

Add your OpenAI API key to keys.sh

```bash
source keys.sh
```

Confirm that you have set your environment variable

```bash
echo $OPENAI_API_KEY
```


Start Chatbot

```bash
python tools.py
```