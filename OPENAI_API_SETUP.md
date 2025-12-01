# OpenAI API Key Setup Guide

This guide explains how to configure the OpenAI API key for better LLM explanations in the Alzheimer's Disease Classification system.

## Why Use OpenAI API?

- **Better Explanations**: GPT-4o-mini provides more natural and detailed explanations
- **Faster**: No need to download and load local models (~1GB)
- **More Reliable**: Consistent quality explanations

## Step 1: Get Your API Key

1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in to your account
3. Click "Create new secret key"
4. Copy the API key (it starts with `sk-`)
5. **Important**: Save it immediately - you won't be able to see it again!

## Step 2: Set the API Key

Choose **ONE** of the following methods:

### Method 1: Environment Variable (Recommended for Windows PowerShell)

**PowerShell (Current Session Only):**
```powershell
$env:OPENAI_API_KEY="sk-your-actual-api-key-here"
```

**PowerShell (Permanent - User Level):**
```powershell
[System.Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'sk-your-actual-api-key-here', 'User')
```

**PowerShell (Permanent - System Level - Requires Admin):**
```powershell
[System.Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'sk-your-actual-api-key-here', 'Machine')
```

After setting permanently, restart your terminal/PowerShell.

### Method 2: Environment Variable (Windows CMD)

**CMD (Current Session Only):**
```cmd
set OPENAI_API_KEY=sk-your-actual-api-key-here
```

**CMD (Permanent - User Level):**
```cmd
setx OPENAI_API_KEY "sk-your-actual-api-key-here"
```

After using `setx`, close and reopen your terminal.

### Method 3: .env File (Easiest)

1. Create a file named `.env` in the project root directory (`D:\Capstone\.env`)
2. Add this line to the file:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```
3. Replace `sk-your-actual-api-key-here` with your actual API key
4. Save the file

**Note**: The `.env` file is automatically ignored by git (in `.gitignore`) so your key won't be committed.

### Method 4: System Environment Variables (Windows GUI)

1. Press `Win + R`, type `sysdm.cpl`, press Enter
2. Go to "Advanced" tab
3. Click "Environment Variables"
4. Under "User variables" or "System variables", click "New"
5. Variable name: `OPENAI_API_KEY`
6. Variable value: `sk-your-actual-api-key-here`
7. Click OK on all dialogs
8. **Restart your terminal/IDE** for changes to take effect

## Step 3: Verify the Setup

1. Start the Flask app:
   ```powershell
   python app.py
   ```
   or
   ```powershell
   py -3.11 app.py
   ```

2. Look for this message:
   ```
   [OK] LLM Explainer initialized
   ```
   
   If you see:
   ```
   ⚠️  OpenAI API key not found. Using local model fallback.
   ```
   Then the API key is not set correctly. Check the steps above.

## Troubleshooting

### "API key not found" Error

1. **Check if the key is set:**
   ```powershell
   echo $env:OPENAI_API_KEY
   ```
   Should show your API key (starts with `sk-`)

2. **If using .env file:**
   - Make sure the file is named exactly `.env` (with the dot)
   - Make sure it's in the project root directory (`D:\Capstone`)
   - Make sure `python-dotenv` is installed:
     ```powershell
     pip install python-dotenv
     ```

3. **Restart your terminal/IDE** after setting environment variables

### "Invalid API key" Error

- Make sure you copied the entire key (it's long, ~50 characters)
- Make sure there are no extra spaces
- Check that the key starts with `sk-`
- Verify the key is active at [OpenAI Platform](https://platform.openai.com/api-keys)

### API Rate Limits

- Free tier has limited requests per minute
- If you hit rate limits, the app will fall back to local model
- Consider upgrading your OpenAI plan for higher limits

## Cost Information

- **GPT-4o-mini**: Very affordable (~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens)
- Each explanation uses ~200 tokens, so ~$0.0001 per explanation
- Free tier includes $5 credit to start

## Security Best Practices

1. **Never commit your API key to git** - The `.env` file is already in `.gitignore`
2. **Don't share your API key** - Treat it like a password
3. **Rotate keys regularly** - If exposed, regenerate it immediately
4. **Set usage limits** - Go to OpenAI dashboard to set spending limits

## Without API Key

If you don't set an API key, the system will:
- Use a local model (google/flan-t5-base) which downloads automatically
- Work slower but still provide explanations
- Require ~1GB disk space for the local model

---

**Need Help?** Check the main `README.md` or `SETUP_INSTRUCTIONS.md` for more information.

